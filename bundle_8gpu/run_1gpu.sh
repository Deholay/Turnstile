#!/bin/bash
# The 1-GPU job: vLLM stages (recompute time t(n), end-to-end recompute-front check).
# Needs VLLM_PY and MODEL_8B in config.env. Uses only the first GPU.
cd "$(dirname "$0")"
export GPUS_OVERRIDE=${GPUS_OVERRIDE:-0}
bash scripts/01_build.sh || { echo "build failed"; exit 1; }
bash scripts/30_prefill.sh || echo "stage 30 failed (continuing)"
bash scripts/40_e2e_front.sh || echo "stage 40 failed"
python3 scripts/summarize.py out > /dev/null
