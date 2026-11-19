"""P2: end-to-end TTFT and greedy-output equality for KvrSSDConnector in vLLM 0.10.1.1.

Same method as killtest/lmc_ttft.py: 32,768 random history tokens + 128 new tokens,
TTFT = wall time of llm.generate(max_tokens=1), 3 warm-up + 5 measured iterations.
One engine serves all phases; the recompute reference opts out per request (kvr_skip).

  python ttft_kvr.py OUT.json        (env: KT_MODEL, KVR_DIR, KVR_LIB, KVR_LOG)
"""
import glob
import json
import os
import random
import statistics as st
import sys
import time

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.inputs import TokensPrompt

out = sys.argv[1]
M = os.environ["KT_MODEL"]
D = os.environ["KVR_DIR"]
LOG = os.environ["KVR_LOG"]
HIST, NEW, GEN = 32768, 128, 32
extra = dict(kvr_dir=D, kvr_lib=os.environ["KVR_LIB"], kvr_log=LOG,
             kvr_threads=int(os.environ.get("KVR_THREADS", 11)),
             kvr_qd=int(os.environ.get("KVR_QD", 32)))
llm = LLM(model=M, max_model_len=33024, max_num_batched_tokens=33024,
          gpu_memory_utilization=0.92, enable_prefix_caching=False,
          kv_transfer_config=KVTransferConfig(
              kv_connector="KvrSSDConnector", kv_role="kv_both",
              kv_connector_module_path="kvr_connector", kv_connector_extra_config=extra))


def sp(max_tokens, **kvp):
    return SamplingParams(max_tokens=max_tokens, temperature=0,
                          extra_args={"kv_transfer_params": kvp} if kvp else None)


def gen(tokens, params):
    t = time.perf_counter()
    o = llm.generate([TokensPrompt(prompt_token_ids=tokens)], params, use_tqdm=False)
    return time.perf_counter() - t, list(o[0].outputs[0].token_ids)


def last_log(event):
    with open(LOG) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    return [r for r in recs if r["event"] == event][-1]


def clean():
    for f in glob.glob(os.path.join(D, "*")):
        os.remove(f)


rng = random.Random(0)
rows = []
# --- timing: kvr load vs recompute, interleaved on the same prompts
for i in range(3 + 5):
    hist = [rng.randrange(1000, 120000) for _ in range(HIST)]
    new = [rng.randrange(1000, 120000) for _ in range(NEW)]
    store_s, _ = gen(hist, sp(1))
    last_log("store")
    ttft_kvr, _ = gen(hist + new, sp(1))
    ld = last_log("load")
    ttft_rec, _ = gen(hist + new, sp(1, kvr_skip=True))
    rows.append(dict(iter=i, warmup=i < 3, store_req_s=store_s, ttft_kvr_s=ttft_kvr,
                     ttft_recompute_s=ttft_rec, load_wall_s=ld["wall_s"], pipe_s=ld["pipe_s"],
                     read_s=ld["read_s"], tokens=ld["tokens"]))
    print("ROW", json.dumps(rows[-1]), flush=True)
    clean()

# --- correctness: per-layer checksums of restored KV + greedy output equality
checks = []
for i in range(2):
    hist = [rng.randrange(1000, 120000) for _ in range(HIST)]
    new = [rng.randrange(1000, 120000) for _ in range(NEW)]
    gen(hist, sp(1, kvr_verify=True))
    _, ref = gen(hist + new, sp(GEN, kvr_skip=True))
    _, got = gen(hist + new, sp(GEN, kvr_verify=True))
    ld = last_log("load")
    first_diff = next((k for k, (a, b) in enumerate(zip(ref, got)) if a != b), None)
    checks.append(dict(prompt=i, kv_checksum_ok=ld.get("verify_ok"), greedy_equal=ref == got,
                       first_diff=first_diff, ref=ref, got=got))
    print("CHECK", json.dumps({k: v for k, v in checks[-1].items() if k not in ("ref", "got")}),
          flush=True)
    clean()

m = [r for r in rows if not r["warmup"]]
summ = {k: st.median(r[k] for r in m) for k in ("ttft_kvr_s", "ttft_recompute_s", "load_wall_s",
                                               "pipe_s", "read_s")}
q = st.quantiles([r["ttft_kvr_s"] for r in m], n=4)
summ["ttft_kvr_iqr"] = q[2] - q[0]
json.dump(dict(model=M, rows=rows, checks=checks, median=summ), open(out, "w"), indent=1)
print("MEDIAN", json.dumps(summ))
