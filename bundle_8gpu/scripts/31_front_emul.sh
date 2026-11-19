#!/bin/bash
# Stage 31: SRPT + recompute-front at the I/O level. Restore i starts its (partial) load of
# [k_i, N) at time t(k_i) = the measured recompute time of its first k_i tokens; the GPU itself
# is idle during that delay (recompute is emulated by the delay, not executed). Compare with
# fair and srpt (load only) from stage 20. Schedule from front_schedule.py.
source "$(dirname "$0")/lib.sh"
stage_begin 31_front_emul
G=$NGPU; [ $G -gt 8 ] && G=8
[ -f "$OUT_DIR/20_concurrent/concurrent.csv" ] || { stage_end 31_front_emul skip "stage 20 results missing"; exit 0; }
PF="$OUT_DIR/30_prefill/prefill.csv"; [ -f "$PF" ] || PF=none
python3 "$B/scripts/front_schedule.py" --concurrent "$OUT_DIR/20_concurrent/concurrent.csv" \
  --prefill "$PF" --G $G --out "$SDIR/schedule.json" | tee "$SDIR/schedule.txt" ||
  { stage_end 31_front_emul fail "front_schedule.py failed"; exit 1; }
CSV="$SDIR/front.csv"
start_sampler; trap stop_sampler EXIT
ok=1
for geo in 8b 70b; do
  python3 - "$SDIR/schedule.json" "$SDIR/spec_${geo}.json" "$DATA_DIR" "$geo" "${GPU_ARR[@]}" <<'EOF' || continue
import json, sys
sch, p, d, geo, *gpus = sys.argv[1:]
s = json.load(open(sch)).get(geo)
if not s:
    sys.exit(1)
json.dump([{"gpu": int(gpus[i]), "file": f"{d}/{geo}_{i}.bin", "geo": geo,
            "start_tokens": s["k"][i], "delay_s": s["delay_s"][i]} for i in range(s["G"])],
          open(p, "w"))
EOF
  python3 "$B/scripts/launch.py" --spec "$SDIR/spec_${geo}.json" --csv "$CSV" \
    --exp front_$geo --reps $REPS --policy srpt --restore "$BIN/restore" \
    --threads $KVR_THREADS --qd $KVR_QD --mode $KVR_MODE --logdir "$SDIR/logs" ||
    { echo "launch failed: $geo" | tee -a "$SDIR/errors.txt"; ok=0; }
done
[ $ok = 1 ] && stage_end 31_front_emul ok || stage_end 31_front_emul fail "see errors.txt"
