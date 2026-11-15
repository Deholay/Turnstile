#!/usr/bin/env python3
"""CPU-only checks (no GPU): cache parsing and KV formula on a tiny Llama, and summarize.py math.

Run with CUDA_VISIBLE_DEVICES= so nothing touches the shared GPU.
"""
import csv
import json
import os
import subprocess
import sys
import tempfile

import torch
from transformers import LlamaConfig, LlamaForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from prefill_bench import cache_tensors, kv_formula_bytes  # noqa: E402


def test_kv_bytes_match_formula():
    # GQA shape like Llama-3 (H_kv < H), shrunk.
    cfg = LlamaConfig(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=3,
                      num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=256)
    model = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
    n = 37  # odd length on purpose
    with torch.inference_mode():
        out = model(input_ids=torch.randint(0, 512, (1, n)), use_cache=True, logits_to_keep=1)
    tensors = list(cache_tensors(out.past_key_values))
    assert len(tensors) == 2 * cfg.num_hidden_layers
    assert all(t.shape == (1, 2, n, 16) for t in tensors), [t.shape for t in tensors]
    actual = sum(t.numel() * t.element_size() for t in tensors)
    assert actual == kv_formula_bytes(cfg, n, 2) == 2 * 3 * 2 * 16 * 2 * n
    assert out.logits.shape[1] == 1, "logits_to_keep=1 not honored"


def test_llama31_8b_formula_is_128kib():
    cfg = LlamaConfig(hidden_size=4096, num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8)
    assert kv_formula_bytes(cfg, 1, 2) == 128 * 1024


def test_summarize_verdict():
    with tempfile.TemporaryDirectory() as d:
        for i, bw in enumerate([6e9, 7e9, 8e9], start=4):
            json.dump({"jobs": [{"read": {"bw_bytes": bw}}]}, open(f"{d}/fio-run-{i}.json", "w"))
        json.dump({"fio": "ignored"}, open(f"{d}/fio-warmup-1.json", "w"))  # warm-up must be ignored
        json.dump({"h2d_pinned_bytes_per_s": 50e9, "kv_bytes_per_token_formula": 131072},
                  open(f"{d}/prefill_meta.json", "w"))
        with open(f"{d}/prefill.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tokens", "phase", "iter", "wall_s", "kv_bytes", "kv_bytes_formula", "peak_alloc_bytes"])
            w.writerow([1000, "warmup", 0, 99.0, 7_000_000_000, 0, 0])  # must be ignored
            for i, s in enumerate([2.0, 2.0, 2.0]):  # read 1.0 s / prefill 2.0 s -> 0.5 -> FAIL (strict <)
                w.writerow([1000, "run", i + 1, s, 7_000_000_000, 0, 0])
            for i, s in enumerate([3.0, 3.0, 3.0]):  # 1.0 / 3.0 -> PASS
                w.writerow([2000, "run", i + 1, s, 7_000_000_000, 0, 0])
        subprocess.run([sys.executable, os.path.join(os.path.dirname(HERE), "summarize.py"), d],
                       check=True, capture_output=True)
        rows = list(csv.DictReader(open(f"{d}/summary.tsv"), delimiter="\t"))
        assert [r["verdict"] for r in rows] == ["FAIL", "PASS"], rows
        assert rows[0]["nvme_read_s"] == "1.0000" and rows[0]["prefill_median_s"] == "2.0000"
        assert rows[0]["read_plus_h2d_s"] == "1.1400"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
