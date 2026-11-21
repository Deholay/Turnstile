# P6 copy of killtest/lmc_ttft.py, WARMUP/RUNS env added.
import os, sys, time, random, json, glob, statistics as st
mode = sys.argv[1]            # "lmcache" or "recompute"
out = sys.argv[2]
disk = os.environ.get("KT_DISK_DIR", "")
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
M = os.environ["KT_MODEL"]
kw = dict(model=M, max_model_len=33024, gpu_memory_utilization=0.92, enable_prefix_caching=False)
if mode == "lmcache":
    from vllm.config import KVTransferConfig
    kw["kv_transfer_config"] = KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both")
llm = LLM(**kw)
sp = SamplingParams(max_tokens=1, temperature=0)
rng = random.Random(0)
HIST, NEW = 32768, 128

def files():
    return [f for f in glob.glob(os.path.join(disk, "**", "*"), recursive=True) if os.path.isfile(f)]

def drop_cache():
    n = 0
    for f in files():
        fd = os.open(f, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd); n += 1
    return n

def wait_flush(timeout=120):
    last, stable, t0 = -1, 0, time.time()
    while time.time() - t0 < timeout:
        s = sum(os.path.getsize(f) for f in files())
        stable = stable + 1 if s == last else 0
        if stable >= 3:
            return s
        last = s; time.sleep(1)
    return last

rows = []
W, N = int(os.environ.get("WARMUP", 3)), int(os.environ.get("RUNS", 5))
for i in range(W + N):
    hist = [rng.randrange(1000, 120000) for _ in range(HIST)]
    new = [rng.randrange(1000, 120000) for _ in range(NEW)]
    store_s = flush_bytes = nfiles = None
    if mode == "lmcache":
        t = time.perf_counter(); llm.generate([TokensPrompt(prompt_token_ids=hist)], sp, use_tqdm=False); store_s = time.perf_counter() - t
        os.sync(); flush_bytes = wait_flush(); nfiles = drop_cache()
    t = time.perf_counter(); llm.generate([TokensPrompt(prompt_token_ids=hist + new)], sp, use_tqdm=False); ttft = time.perf_counter() - t
    rows.append(dict(iter=i, warmup=i < W, mode=mode, ttft_s=ttft, store_req_s=store_s, disk_bytes=flush_bytes, files_dropped=nfiles))
    print("ROW", json.dumps(rows[-1]), flush=True)
    if mode == "lmcache":
        for f in files():
            os.remove(f)
v = sorted(r["ttft_s"] for r in rows if not r["warmup"]); q = st.quantiles(v, n=4)
json.dump(dict(rows=rows, median=st.median(v), iqr=q[2] - q[0]), open(out, "w"), indent=1)
print("MEDIAN", mode, st.median(v), "IQR", q[2] - q[0])
