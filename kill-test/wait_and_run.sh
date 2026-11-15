#!/usr/bin/env bash
# Wait until the shared GPU has had no compute process for IDLE_CHECKS consecutive polls, then run the kill test.
# run_kill_test.sh re-checks the GPU right before each phase.
set -euo pipefail

POLL_S=${POLL_S:-30}
IDLE_CHECKS=${IDLE_CHECKS:-10}   # 10 x 30 s = 5 min of continuous idle
HERE=$(cd "$(dirname "$0")" && pwd)

idle=0
while (( idle < IDLE_CHECKS )); do
  if [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]; then
    idle=$((idle + 1))
  else
    idle=0
  fi
  echo "$(date -Is) idle_checks=$idle/$IDLE_CHECKS"
  (( idle < IDLE_CHECKS )) && sleep "$POLL_S"
done
echo "$(date -Is) GPU idle for $((POLL_S * IDLE_CHECKS)) s, starting kill test"
exec "$HERE/run_kill_test.sh"
