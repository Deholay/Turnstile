"""P3-dev: recompute-front + suffix load through KvrSSDConnector, two requests per restore.

A queued GPU recomputes history[0:k) while it waits for the SSD array, then loads only
[k, N). With vLLM's own interface this is two requests on one engine with prefix caching:

  req1  hist[:k], max_tokens=1, kvr_skip    recompute the front; its blocks stay cached
  req2  hist+new                            prefix-cache hit k -> connector loads [k, N)

Per prompt:
  1. reset; hist (kvr_verify)               prefill hist, store it to SSD
  2. hist+new, kvr_skip                     reference: plain prefix-cache path (hist on GPU)
  3. per k: reset; req1; req2 (kvr_verify)  check (a) load start == k, only N-k tokens and
                                            N/16-k/16 blocks loaded, loaded segments match
                                            the stored checksums; (b) greedy output == 2
  4. per k: reset; req1; req2 max_tokens=1  timing (no verify): t_req1, t_req2, load wall,
     and t_ref1 = hist+new max_tokens=1 with hist cached on GPU (no load).
     req2 overhead = t_req2 - load_wall - t_ref1 (scheduling/launch cost of the split).
k = 0 skips req1 (whole-file load, the P3a path).

  python check_front.py OUT.json   (env: KT_MODEL, KVR_DIR, KVR_LIB, KVR_LOG;
                                    optional KF_HIST, KF_KS="0,8192,8240,16256", KF_GPU_UTIL)
disp (RTX 5090): correctness and behavior only; its SSDs are far slower than the H200 nodes.
"""
import json
import os
import random
import sys
import time

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.inputs import TokensPrompt

out = sys.argv[1]
M = os.environ["KT_MODEL"]
D = os.environ["KVR_DIR"]
LOG = os.environ["KVR_LOG"]
HIST = int(os.environ.get("KF_HIST", 16384))
NEW, GEN, PROMPTS = 128, 64, int(os.environ.get("KF_PROMPTS", 2))
KS = [int(x) for x in os.environ.get(
    "KF_KS", f"0,{HIST // 2},{HIST // 2 + 48},{HIST - 128}").split(",")]
extra = dict(kvr_dir=D, kvr_lib=os.environ["KVR_LIB"], kvr_log=LOG, kvr_threads=8, kvr_qd=16)
MAXLEN = HIST + NEW + GEN + 128
llm = LLM(model=M, max_model_len=MAXLEN, max_num_batched_tokens=MAXLEN,
          gpu_memory_utilization=float(os.environ.get("KF_GPU_UTIL", 0.85)),
          enable_prefix_caching=True,
          kv_transfer_config=KVTransferConfig(
              kv_connector="KvrSSDConnector", kv_role="kv_both",
              kv_connector_module_path="kvr_connector", kv_connector_extra_config=extra))


def gen(tokens, max_tokens, **kvp):
    p = SamplingParams(max_tokens=max_tokens, temperature=0,
                       extra_args={"kv_transfer_params": kvp} if kvp else None)
    t0 = time.perf_counter()
    o = llm.generate([TokensPrompt(prompt_token_ids=tokens)], p, use_tqdm=False)[0]
    return list(o.outputs[0].token_ids), o.num_cached_tokens, time.perf_counter() - t0


def reset():
    # vLLM 0.10.1.1 returns None here through the MP client, so the return value says nothing.
    # A reset that did not take effect shows up in the next request: req2 then hits the old
    # GPU blocks and its load record is missing or has the wrong start (suffix_ok / timing_ok).
    llm.reset_prefix_cache()


def loads():
    if not os.path.exists(LOG):
        return []
    with open(LOG) as f:
        return [r for r in map(json.loads, filter(str.strip, f)) if r["event"] == "load"]


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def front(hist, new, k, max_tokens, **kvp):
    """req1 (unless k == 0) then req2; returns req2 output, its load record, timings."""
    reset()
    t1 = None
    if k:
        _, _, t1 = gen(hist[:k], 1, kvr_skip=True)
    n0 = len(loads())
    tg = time.perf_counter()
    got, cached, t2 = gen(hist + new, max_tokens, **kvp)
    ld = loads()[n0:]
    return got, cached, ld, t1, t2, tg


rng = random.Random(1)
blk = llm.llm_engine.cache_config.block_size
checks = []
for i in range(PROMPTS):
    hist = [rng.randrange(1000, 120000) for _ in range(HIST)]
    new = [rng.randrange(1000, 120000) for _ in range(NEW)]
    reset()
    gen(hist, 1, kvr_verify=True)  # store
    ref_pc, cached_pc, _ = gen(hist + new, GEN, kvr_skip=True)
    t_ref1 = min(gen(hist + new, 1, kvr_skip=True)[2] for _ in range(3))
    for k in KS:
        got, cached, ld, _, _, _ = front(hist, new, k, GEN, kvr_verify=True)
        exp_start = k // blk * blk
        r = ld[0] if len(ld) == 1 else {}
        suffix_ok = (len(ld) == 1 and r["start"] == exp_start
                     and r["loaded_tokens"] == HIST - exp_start
                     and r["blocks"] == (HIST - exp_start) // blk and r.get("verify_ok") is True)
        # timing pass, no verify; 3 repeats, keep the median of req2
        reps, timing_ok = [], True
        for _ in range(3):
            _, _, ldt, t1, t2, _ = front(hist, new, k, 1)
            timing_ok &= len(ldt) == 1 and ldt[0]["start"] == exp_start
            reps.append((t2, t1, ldt[0]["wall_s"] if ldt else None,
                         ldt[0]["pipe_s"] if ldt else None))
        reps.sort(key=lambda x: x[0])
        t2, t1, lw, lp = reps[1]
        c = dict(prompt=i, k=k, num_cached_tokens=cached, loads=len(ld),
                 load_start=r.get("start"), loaded_tokens=r.get("loaded_tokens"),
                 load_blocks=r.get("blocks"), verify_ok=r.get("verify_ok"),
                 prefix_seg_mismatch=r.get("prefix_seg_mismatch"),
                 prefix_seg_total=r.get("prefix_seg_total"), suffix_ok=suffix_ok,
                 timing_ok=timing_ok,
                 equal_prefix_cache=got == ref_pc, first_diff=first_diff(got, ref_pc),
                 t_req1_s=t1, t_req2_s=t2, load_wall_s=lw, load_pipe_s=lp, t_ref1_s=t_ref1,
                 req2_overhead_s=(t2 - lw - t_ref1) if lw is not None else None,
                 got=got, ref=ref_pc)
        checks.append(c)
        print("CHECK", json.dumps({x: v for x, v in c.items() if x not in ("got", "ref")}),
              flush=True)
    for f in os.listdir(D):
        os.remove(os.path.join(D, f))

ok = all(c["suffix_ok"] and c["timing_ok"] and c["equal_prefix_cache"] for c in checks)
json.dump(dict(model=M, hist=HIST, new=NEW, gen=GEN, ks=KS, block=blk, checks=checks, pass_=ok),
          open(out, "w"), indent=1)
print("RESULT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
