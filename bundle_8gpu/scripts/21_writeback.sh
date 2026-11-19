#!/bin/bash
# Stage 21: same as stage 20 at G = all GPUs, while a background writer (fio, rate-limited to
# WB MB/s, separate files) emulates KV write-back of other sessions. Skipped without fio.
source "$(dirname "$0")/lib.sh"
stage_begin 21_writeback
command -v fio >/dev/null || { stage_end 21_writeback skip "fio not installed"; exit 0; }
WB=${WB_RATE_MBS:-$(cat "$OUT_DIR/10_fio/writeback_rate_MBs.txt" 2>/dev/null)}
[ -n "$WB" ] || { stage_end 21_writeback skip "no write rate (stage 10 missing and WB_RATE_MBS empty)"; exit 0; }
CSV="$SDIR/writeback.csv"
G=$NGPU; [ $G -gt 8 ] && G=8
W="$DATA_DIR/wb"; mkdir -p "$W"
start_sampler
fio --name=wb --rw=write --bs=1M --iodepth=8 --numjobs=2 --size=8G --time_based --runtime=7200 \
    --rate=$((WB / 2))m --directory=$W --direct=1 --ioengine=libaio --group_reporting \
    --output-format=json --output="$SDIR/writer.json" > /dev/null 2>&1 & FIO=$!
trap 'kill $FIO 2>/dev/null; wait $FIO 2>/dev/null; stop_sampler; rm -rf "$W"' EXIT
sleep 10   # let the writer reach its rate
ok=1
for geo in 8b 70b; do
  python3 - "$SDIR/spec_${geo}.json" "$DATA_DIR" "$geo" "$G" "${GPU_ARR[@]}" <<'EOF'
import json, sys
p, d, geo, G, *gpus = sys.argv[1:]
json.dump([{"gpu": int(gpus[i]), "file": f"{d}/{geo}_{i}.bin", "geo": geo}
           for i in range(int(G))], open(p, "w"))
EOF
  for pol in fair srpt; do
    python3 "$B/scripts/launch.py" --spec "$SDIR/spec_${geo}.json" --csv "$CSV" \
      --exp writeback_$geo --reps $REPS --policy $pol --restore "$BIN/restore" \
      --threads $KVR_THREADS --qd $KVR_QD --mode $KVR_MODE --logdir "$SDIR/logs" ||
      { echo "launch failed: $geo $pol" | tee -a "$SDIR/errors.txt"; ok=0; }
  done
done
echo "$WB" > "$SDIR/writeback_rate_MBs.txt"
[ $ok = 1 ] && stage_end 21_writeback ok "writer ${WB} MB/s" || stage_end 21_writeback fail "see errors.txt"
