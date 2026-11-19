#!/bin/bash
# Stage 30: measure recompute cost t(n) with vLLM on one GPU, for 8B (required for stage 31)
# and 70B (only if the GPU has >= 120 GB). Needs VLLM_PY and MODEL_8B.
source "$(dirname "$0")/lib.sh"
stage_begin 30_prefill
[ -n "$VLLM_PY" ] && [ -n "$MODEL_8B" ] || { stage_end 30_prefill skip "VLLM_PY or MODEL_8B not set"; exit 0; }
g0=${GPU_ARR[0]}
NP=$(numa_prefix $g0)
CUDA_VISIBLE_DEVICES=$g0 $NP "$VLLM_PY" "$B/scripts/prefill_vllm.py" --model "$MODEL_8B" \
  --tag 8b --out "$SDIR/prefill.csv" > "$SDIR/prefill_8b.log" 2>&1 ||
  { stage_end 30_prefill fail "8B prefill failed, see prefill_8b.log"; exit 1; }
note="8b ok"
mem=$(nvidia-smi -i $g0 --query-gpu=memory.total --format=csv,noheader,nounits | tr -dc 0-9)
if [ -n "$MODEL_70B" ] && [ "${mem:-0}" -ge 120000 ]; then
  CUDA_VISIBLE_DEVICES=$g0 $NP "$VLLM_PY" "$B/scripts/prefill_vllm.py" --model "$MODEL_70B" \
    --tag 70b --out "$SDIR/prefill.csv" --gpu-util 0.95 > "$SDIR/prefill_70b.log" 2>&1 &&
    note="$note; 70b ok" || note="$note; 70b FAILED (see prefill_70b.log)"
else
  note="$note; 70b skipped (MODEL_70B empty or GPU < 120 GB)"
fi
stage_end 30_prefill ok "$note"
