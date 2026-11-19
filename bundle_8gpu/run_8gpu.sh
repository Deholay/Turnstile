#!/bin/bash
# The 8-GPU job: all multi-GPU stages, in priority order, inside ONE allocation that holds all
# GPUs. Budget-aware: set TIME_LIMIT_MIN to the allocation's time limit (default 23); a stage is
# skipped when the time left is below its estimate, and 99_pack always runs (also on SIGUSR1 /
# SIGTERM from the scheduler). Run run_1gpu.sh FIRST if possible, so stage 31 can use the
# measured recompute times t(n).
cd "$(dirname "$0")"
T_END=$(( $(date +%s) + ${TIME_LIMIT_MIN:-23} * 60 - 120 ))   # keep 2 min for packing
CHILD=
pack() { bash scripts/99_pack.sh; }
on_signal() {
  echo "signal received: stopping stage and packing" | tee -a out/run.log
  [ -n "$CHILD" ] && pkill -TERM -P "$CHILD" 2>/dev/null; [ -n "$CHILD" ] && kill "$CHILD" 2>/dev/null
  pack; exit 1
}
trap on_signal USR1 TERM
run() { bash "scripts/$1.sh" & CHILD=$!; wait $CHILD; local rc=$?; CHILD=; return $rc; }
left() { echo $(( (T_END - $(date +%s)) / 60 )); }
mkdir -p out
for s in 00_preflight 01_build 02_smoke; do
  run $s || { echo "gate stage $s failed; stop and see running_guide.md"; pack; exit 1; }
done
# stage:estimated minutes on an 8-GPU H200-class node, highest priority first
for item in 20_concurrent:7 31_front_emul:2 21_writeback:3 32_arrivals:5 10_fio:2; do
  s=${item%%:*}; need=${item##*:}
  if [ "$(left)" -lt "$need" ]; then
    echo "skipping $s: $(left) min left, needs ~$need" | tee -a out/run.log; continue
  fi
  run $s || echo "stage $s failed (continuing)"
done
pack
