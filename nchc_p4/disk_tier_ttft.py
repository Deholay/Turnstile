"""P4-A: TTFT of vLLM 0.32's native disk tier (SimpleCPUOffloadConnector, kv_offload_backend=disk).

The disk backend (vllm/v1/simple_kv_offload/disk_backend.py) moves blocks GPU <-> NVMe through
pinned staging buffers with O_DIRECT preadv/pwritev (page cache off by default), one load thread
and one store thread, double-buffered. Stores are eager (every computed block is written).

Per prompt (history HIST random tokens + NEW new tokens, distinct prompts):
  store phase: generate(hist, max_tokens=1)  -> blocks computed and stored to disk
  wait SETTLE seconds for the async stores to drain
  load phase:  reset_prefix_cache(); generate(hist+new, max_tokens=1), timed
               -> hist comes from the disk tier, new is prefilled; wall = TTFT
num_cached_tokens of each load output is recorded: it must be ~HIST, otherwise the load did not
come from disk (store not finished or evicted) and that run is flagged.

  python disk_tier_ttft.py --model DIR --root DIR --out OUT.json [--warmup 2 --runs 4]
"""
import argparse
import json
import os
import random
import statistics
import time

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--hist", type=int, default=32768)
ap.add_argument("--new", type=int, default=128)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--runs", type=int, default=4)
ap.add_argument("--disk-gib", type=int, default=72)
ap.add_argument("--settle", type=float, default=30.0)
ap.add_argument("--gpu-util", type=float, default=0.92)
a = ap.parse_args()

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.config import KVTransferConfig  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

extra = dict(kv_offload_backend="disk", disk_path=os.path.join(a.root, "kv"),
             disk_capacity_bytes=a.disk_gib << 30)
t = time.perf_counter()
llm = LLM(model=a.model, max_model_len=a.hist + a.new + 256,
          max_num_batched_tokens=a.hist + a.new + 256,
          enable_prefix_caching=True, gpu_memory_utilization=a.gpu_util,
          kv_transfer_config=KVTransferConfig(kv_connector="SimpleCPUOffloadConnector",
                                              kv_role="kv_both",
                                              kv_connector_extra_config=extra))
load_model_s = time.perf_counter() - t
print(f"engine ready in {load_model_s:.1f} s", flush=True)
sp = SamplingParams(max_tokens=1, temperature=0)
rng = random.Random(0)
n = a.warmup + a.runs
P = [([rng.randrange(1000, 120000) for _ in range(a.hist)],
      [rng.randrange(1000, 120000) for _ in range(a.new)]) for _ in range(n)]


def gen(ids):
    t0 = time.perf_counter()
    o = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)[0]
    return time.perf_counter() - t0, o.num_cached_tokens


store = []
for i, (h, _) in enumerate(P):
    w, _c = gen(h)
    store.append(w)
    print(f"store {i}: prefill+store-issue {w:.3f} s", flush=True)
time.sleep(a.settle)

loads = []
for i, (h, nw) in enumerate(P):
    assert llm.reset_prefix_cache(), "reset_prefix_cache failed"
    w, c = gen(h + nw)
    ok = c >= a.hist - 16
    loads.append(dict(i=i, phase="warmup" if i < a.warmup else "run", ttft_s=w,
                      num_cached_tokens=c, from_disk=ok))
    print(f"load {i}: ttft {w:.3f} s cached {c} from_disk={ok}", flush=True)

runs = [x["ttft_s"] for x in loads if x["phase"] == "run" and x["from_disk"]]
res = dict(model=a.model, hist=a.hist, new=a.new, connector="SimpleCPUOffloadConnector",
           extra=extra, load_model_s=load_model_s, store_wall_s=store, settle_s=a.settle,
           loads=loads, ttft_median_s=statistics.median(runs) if runs else None,
           runs_from_disk=len(runs))
json.dump(res, open(a.out, "w"), indent=1)
print(f"RESULT disk-tier TTFT median {res['ttft_median_s']} s over {len(runs)} runs from disk",
      flush=True)
