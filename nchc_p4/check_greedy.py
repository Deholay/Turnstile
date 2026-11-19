"""P3a: greedy-output equality of the KvrSSDConnector load path against vLLM prefix caching.

P2 compared the load path with a full recompute of hist+new and saw divergence although the
restored KV was byte-exact: the stored KV comes from a hist-only prefill, the full recompute
computes hist and new together, so the KV differs numerically. The right reference reuses the
same hist-only KV on the GPU — vLLM's own prefix cache. Per prompt, one engine with prefix
caching on:

  1. hist                      prefill hist, store it to SSD (kvr), keep its blocks cached on GPU
  2. hist+new, kvr_skip        reference: GPU prefix-cache hit on hist, prefill new, decode GEN
  3. reset_prefix_cache()
  4. hist+new                  kvr load of hist (checksum-verified), prefill new, decode GEN
  5. reset_prefix_cache(); hist+new, kvr_skip   full recompute (informational, as in P2)

Pass condition: 4 == 2 token for token, for every prompt.

  python check_greedy.py OUT.json        (env: KT_MODEL, KVR_DIR, KVR_LIB, KVR_LOG)
"""
import glob
import json
import os
import random
import sys

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.inputs import TokensPrompt

out = sys.argv[1]
M = os.environ["KT_MODEL"]
D = os.environ["KVR_DIR"]
LOG = os.environ["KVR_LOG"]
HIST, NEW, GEN, PROMPTS = 32768, 128, 64, 3
extra = dict(kvr_dir=D, kvr_lib=os.environ["KVR_LIB"], kvr_log=LOG, kvr_threads=11, kvr_qd=32)
llm = LLM(model=M, max_model_len=33024, max_num_batched_tokens=33024,
          gpu_memory_utilization=0.92, enable_prefix_caching=True,
          kv_transfer_config=KVTransferConfig(
              kv_connector="KvrSSDConnector", kv_role="kv_both",
              kv_connector_module_path="kvr_connector", kv_connector_extra_config=extra))


def gen(tokens, max_tokens, **kvp):
    p = SamplingParams(max_tokens=max_tokens, temperature=0,
                       extra_args={"kv_transfer_params": kvp} if kvp else None)
    o = llm.generate([TokensPrompt(prompt_token_ids=tokens)], p, use_tqdm=False)[0]
    return list(o.outputs[0].token_ids), o.num_cached_tokens


def reset():
    # vLLM 0.10.1.1's LLM.reset_prefix_cache() returns None through the MP client (not a bool),
    # so its return value cannot be asserted. Whether the reset took effect is checked by the
    # recorded pc_cached_tokens / kvr_loaded_tokens of the following requests instead.
    llm.reset_prefix_cache()


def loads():
    if not os.path.exists(LOG):
        return []
    with open(LOG) as f:
        return [r for r in map(json.loads, filter(str.strip, f)) if r["event"] == "load"]


def first_diff(a, b):
    return next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None)


rng = random.Random(1)
checks = []
for i in range(PROMPTS):
    hist = [rng.randrange(1000, 120000) for _ in range(HIST)]
    new = [rng.randrange(1000, 120000) for _ in range(NEW)]
    reset()
    gen(hist, 1, kvr_verify=True)
    ref_pc, cached_pc = gen(hist + new, GEN, kvr_skip=True)
    reset()
    n0 = len(loads())
    got, cached_kvr = gen(hist + new, GEN, kvr_verify=True)
    ld = loads()[n0:]
    reset()
    ref_full, _ = gen(hist + new, GEN, kvr_skip=True)
    c = dict(prompt=i, pc_cached_tokens=cached_pc, kvr_loaded_tokens=sum(r["tokens"] for r in ld),
             kv_checksum_ok=bool(ld) and all(r.get("verify_ok") for r in ld),
             equal_prefix_cache=got == ref_pc, first_diff_prefix_cache=first_diff(got, ref_pc),
             equal_full_recompute=got == ref_full,
             first_diff_full_recompute=first_diff(got, ref_full),
             got=got, ref_prefix_cache=ref_pc, ref_full_recompute=ref_full)
    checks.append(c)
    print("CHECK", json.dumps({k: v for k, v in c.items() if not k.startswith(("got", "ref"))}),
          flush=True)
    for f in glob.glob(os.path.join(D, "*")):
        os.remove(f)

ok = all(c["equal_prefix_cache"] and c["kv_checksum_ok"] and c["kvr_loaded_tokens"] == HIST
         and c["pc_cached_tokens"] == HIST for c in checks)
json.dump(dict(model=M, hist=HIST, new=NEW, gen=GEN, checks=checks, pass_=ok), open(out, "w"),
          indent=1)
print("RESULT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
