#!/usr/bin/env bash
# Small-model kill test on disp: fio sequential read of the data NVMe + HF prefill timing.
# Refuses to start a phase while any other process is on the GPU (shared lab machine).
set -euo pipefail

D=/data/personal/r15942056
VENV=$D/venvs/kvssd
FIO=$D/tools/bin/fio
MODEL=${MODEL:-$D/models/Llama-3.1-8B-Instruct}
LENGTHS=${LENGTHS:-"4096 8192 16384 32768"}
WARMUP=${WARMUP:-3}
RUNS=${RUNS:-5}
FIO_FILE=${FIO_FILE:-$D/kvtest/fio-seqread-8g.dat}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$D/results/killtest-$(date +%Y%m%d-%H%M%S)}
export HF_HOME=$D/hf HF_HUB_OFFLINE=1
mkdir -p "$OUT"

gpu_apps() { nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; }
require_idle() {
  local apps
  apps=$(gpu_apps)
  if [[ -n $apps ]]; then
    printf 'GPU busy, not starting %s:\n%s\n' "$1" "$apps" | tee -a "$OUT/aborted.txt" >&2
    exit 3
  fi
}
snapshot() {
  {
    echo "== $1 $(date -Is)"
    nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used --format=csv
    gpu_apps
  } >> "$OUT/gpu-snapshots.txt"
}

{
  date -Is; hostname
  nvidia-smi --query-gpu=name,driver_version,pcie.link.gen.current,pcie.link.width.current,clocks.max.sm --format=csv
  sha256sum "$HERE"/*.py "$HERE"/*.sh
  "$VENV/bin/python" -c 'import torch, transformers; print("torch", torch.__version__, "cuda", torch.version.cuda); print("transformers", transformers.__version__)'
  "$FIO" --version
  echo "model $MODEL commit $(head -1 "$MODEL/.cache/huggingface/download/config.json.metadata" 2>/dev/null || echo unknown)"
  findmnt -T "$(dirname "$FIO_FILE")" -o SOURCE,TARGET,FSTYPE   # fio creates FIO_FILE later; findmnt fails on a missing path
  echo "LENGTHS=$LENGTHS WARMUP=$WARMUP RUNS=$RUNS FIO_FILE=$FIO_FILE"
} > "$OUT/env.txt" 2>&1

# Phase 1: NVMe sequential read, proposal settings (bs=1M, QD32, O_DIRECT).
require_idle fio
snapshot fio-before
FIO_ARGS=(--name=seqread --filename="$FIO_FILE" --size=8G --direct=1 --rw=read --bs=1M
          --iodepth=32 --ioengine=io_uring --numjobs=1 --runtime=15 --time_based --output-format=json)
echo "$FIO ${FIO_ARGS[*]}" >> "$OUT/commands.txt"
for i in $(seq 1 $((WARMUP + RUNS))); do
  tag=$([[ $i -le $WARMUP ]] && echo warmup || echo run)
  "$FIO" "${FIO_ARGS[@]}" --output="$OUT/fio-$tag-$i.json"
done
snapshot fio-after

# Phase 2: full prefill (recompute) time and real KV size per history length.
require_idle prefill
snapshot prefill-before
PREFILL=("$VENV/bin/python" "$HERE/prefill_bench.py" --model "$MODEL" --lengths $LENGTHS
         --warmup "$WARMUP" --runs "$RUNS" --out "$OUT")
echo "${PREFILL[*]}" >> "$OUT/commands.txt"
"${PREFILL[@]}" 2>&1 | tee "$OUT/prefill.log"
snapshot prefill-after

"$VENV/bin/python" "$HERE/summarize.py" "$OUT" | tee "$OUT/summary.txt"
echo "results: $OUT"
