#!/usr/bin/env python3
"""Recompute cost t(n): vLLM prefill time of an n-token prompt (1 output token), prefix
caching off, no chunked prefill. Random token ids (dense prefill time does not depend on
values). One GPU (CUDA_VISIBLE_DEVICES). Writes CSV rows: model,tokens,phase,iter,wall_s.

  python prefill_vllm.py --model DIR --out prefill.csv [--lengths ...] [--warmup 2 --runs 5]
"""
import argparse
import csv
import os
import random
import time

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--lengths", type=int, nargs="+",
                default=[1024, 2048, 4096, 6144, 8192, 12288, 16384, 20480, 24576, 28672, 32768])
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--runs", type=int, default=5)
ap.add_argument("--gpu-util", type=float, default=0.90)
# Default: the whole prompt in one batch (as in job 472017). Beyond ~64k tokens vLLM 0.10.1.1
# crashed in the profiling run (illegal memory access, job 481654), so long prompts use chunked
# prefill with this chunk size.
ap.add_argument("--max-batched", type=int, default=0)
a = ap.parse_args()

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

maxlen = max(a.lengths) + 256
llm = LLM(model=a.model, max_model_len=maxlen, max_num_batched_tokens=a.max_batched or maxlen,
          enable_prefix_caching=False, gpu_memory_utilization=a.gpu_util)
sp = SamplingParams(max_tokens=1, temperature=0)
rng = random.Random(0)
fresh = not os.path.exists(a.out)
with open(a.out, "a", newline="") as f:
    w = csv.writer(f)
    if fresh:
        w.writerow(["model", "tokens", "phase", "iter", "wall_s"])
    for n in a.lengths:
        for i in range(a.warmup + a.runs):
            ids = [rng.randrange(1000, 120000) for _ in range(n)]
            t0 = time.perf_counter()
            llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
            wall = time.perf_counter() - t0
            w.writerow([a.tag, n, "warmup" if i < a.warmup else "run", i, f"{wall:.6f}"])
            f.flush()
        print(f"{a.tag} n={n} done", flush=True)
