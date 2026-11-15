#!/usr/bin/env python3
"""Kill test, compute side: time a full prefill of an N-token history and record its KV size.

Recompute cost of a returning session = prefill of its whole history. Token ids are random:
dense-model prefill time does not depend on token values, only on length.
"""
import argparse
import csv
import json
import os
import statistics
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def kv_formula_bytes(cfg, tokens, elem_bytes):
    """Proposal formula: 2 x L x H_kv x d_head x b per token."""
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * elem_bytes * tokens


def cache_tensors(cache):
    """Yield every K and V tensor of a transformers cache object."""
    if hasattr(cache, "layers"):  # transformers >= 4.56 DynamicCache
        for layer in cache.layers:
            yield layer.keys
            yield layer.values
    elif hasattr(cache, "key_cache"):  # older DynamicCache
        yield from cache.key_cache
        yield from cache.value_cache
    else:  # legacy tuple of (k, v) per layer
        for k, v in cache:
            yield k
            yield v


def measure_h2d_bytes_per_s(nbytes=256 << 20, reps=10):
    """Pinned host -> GPU copy bandwidth (the PCIe leg of a naive SSD restore)."""
    host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    for _ in range(2):
        dev.copy_(host, non_blocking=True)
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        dev.copy_(host, non_blocking=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    del host, dev
    torch.cuda.empty_cache()
    return nbytes / statistics.median(times)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="local model directory")
    ap.add_argument("--lengths", type=int, nargs="+", required=True, help="history lengths in tokens")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16
    elem_bytes = torch.finfo(dtype).bits // 8
    cfg = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="sdpa", device_map="cuda"
    ).eval()
    token_high = min(cfg.vocab_size, 100_000)  # stay below Llama-3 special tokens (>= 128000)

    h2d = measure_h2d_bytes_per_s()
    meta = {
        "model": args.model,
        "dtype": str(dtype),
        "attn_implementation": "sdpa",
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads,
        "kv_bytes_per_token_formula": kv_formula_bytes(cfg, 1, elem_bytes),
        "h2d_pinned_bytes_per_s": h2d,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "seed": args.seed,
    }
    with open(os.path.join(args.out, "prefill_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2), flush=True)

    fields = ["tokens", "phase", "iter", "wall_s", "kv_bytes", "kv_bytes_formula", "peak_alloc_bytes"]
    with open(os.path.join(args.out, "prefill.csv"), "w", newline="") as f, torch.inference_mode():
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for n in sorted(args.lengths):
            ids = torch.randint(1000, token_high, (1, n), device="cuda")
            torch.cuda.reset_peak_memory_stats()
            kv_bytes = None
            walls = []
            for i in range(args.warmup + args.runs):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
                torch.cuda.synchronize()
                wall = time.perf_counter() - t0
                if kv_bytes is None:
                    tensors = list(cache_tensors(out.past_key_values))
                    assert all(t.shape[-2] == n for t in tensors), "cache length != history length"
                    assert torch.isfinite(out.logits).all(), "non-finite logits"
                    kv_bytes = sum(t.numel() * t.element_size() for t in tensors)
                del out
                phase = "warmup" if i < args.warmup else "run"
                if phase == "run":
                    walls.append(wall)
                w.writerow({
                    "tokens": n, "phase": phase, "iter": i, "wall_s": f"{wall:.6f}",
                    "kv_bytes": kv_bytes, "kv_bytes_formula": kv_formula_bytes(cfg, n, elem_bytes),
                    "peak_alloc_bytes": torch.cuda.max_memory_allocated(),
                })
                f.flush()
            print(f"tokens={n} prefill median={statistics.median(walls):.4f}s "
                  f"kv={kv_bytes / 2**30:.3f} GiB peak={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
                  flush=True)
            del ids
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
