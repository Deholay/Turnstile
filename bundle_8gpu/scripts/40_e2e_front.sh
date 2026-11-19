#!/bin/bash
# Stage 40: end-to-end recompute-front through vLLM on ONE GPU (8B): correctness of the
# two-request path (req1 = history[:k] recompute, req2 = full prompt, connector loads only
# [k, N)) against vLLM's own prefix-cache path, plus the overhead between the two requests.
source "$(dirname "$0")/lib.sh"
stage_begin 40_e2e_front
[ -n "$VLLM_PY" ] && [ -n "$MODEL_8B" ] || { stage_end 40_e2e_front skip "VLLM_PY or MODEL_8B not set"; exit 0; }
g0=${GPU_ARR[0]}
D="$DATA_DIR/e2e"; mkdir -p "$D"
KT_MODEL="$MODEL_8B" KVR_DIR="$D" KVR_LIB="$BIN/libkvr.so" KVR_LOG="$SDIR/kvr_load.jsonl" \
KF_HIST=32768 KF_KS="0,8192,16384,24576" KF_PROMPTS=2 PYTHONPATH="$B/src" \
CUDA_VISIBLE_DEVICES=$g0 $(numa_prefix $g0) "$VLLM_PY" "$B/src/check_front.py" "$SDIR/check_front.json" \
  > "$SDIR/check_front.log" 2>&1
rc=$?
rm -rf "$D"
grep -E "^CHECK|^RESULT" "$SDIR/check_front.log" | tail -20
[ $rc -eq 0 ] && stage_end 40_e2e_front ok "RESULT PASS" ||
  stage_end 40_e2e_front fail "exit $rc; see check_front.log (RESULT FAIL = correctness problem)"
