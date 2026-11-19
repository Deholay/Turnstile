#!/bin/bash
# Stage 99: summarize and pack OUT_DIR into one tarball next to the bundle. Removes the big
# restore files from DATA_DIR only when called with --clean.
source "$(dirname "$0")/lib.sh"
python3 "$B/scripts/summarize.py" "$OUT_DIR" > /dev/null || echo "summarize.py failed" | tee -a "$OUT_DIR/run.log"
mkdir -p "$OUT_DIR/bundle_changes"
# record any edits the operator made to the bundle (sources and scripts) vs the shipped copy
( cd "$B" && sha256sum src/* scripts/* config.env > "$OUT_DIR/bundle_changes/sha256_now.txt" )
cp "$B/config.env" "$OUT_DIR/bundle_changes/config.env"
T="$B/../kvr_8gpu_results_$(date +%Y%m%d_%H%M).tar.gz"
tar -czf "$T" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")"
echo "packed: $(cd "$(dirname "$T")" && pwd)/$(basename "$T") ($(du -h "$T" | cut -f1))"
if [ "${1:-}" = "--clean" ]; then
  rm -f "$DATA_DIR"/8b_*.bin* "$DATA_DIR"/70b_*.bin* && echo "removed restore files from $DATA_DIR"
fi
