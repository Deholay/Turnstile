#!/bin/bash
# Stage 02: correctness gate. Every check restores into GPU memory and verifies every word.
# Any failure here means later numbers are meaningless: stop and fix (see running_guide.md).
source "$(dirname "$0")/lib.sh"
stage_begin 02_smoke
L="$SDIR/smoke.log"; : > "$L"
g0=${GPU_ARR[0]}
SMALL="$DATA_DIR/smoke_small.bin"
ok=1
run() { echo "+ $*" >> "$L"; "$@" >> "$L" 2>&1; local r=$?; [ $r -eq 0 ] || { echo "FAILED ($r): $*" | tee -a "$L"; ok=0; }; }

# A. small file, both transfer modes, full and partial (start_tokens) restores
run env CUDA_VISIBLE_DEVICES=$g0 "$BIN/restore" --file "$SMALL" --write --layers 4 --tokens 2048
for m in copy direct; do
  for st in 0 1024 1040 1920; do
    run env CUDA_VISIBLE_DEVICES=$g0 "$BIN/restore" --file "$SMALL" --layers 4 --tokens 2048 \
      --runs 1 --warmup 1 --mode $m --start-tokens $st
  done
done
rm -f "$SMALL"

# B. full-size files for 8b geometry (also used by later stages), one restore per GPU alone
ensure_files 8b || ok=0
if [ $ok = 1 ]; then
  for ((i = 0; i < NGPU; i++)); do
    run env CUDA_VISIBLE_DEVICES=${GPU_ARR[$i]} "$BIN/restore" --file "$DATA_DIR/8b_$i.bin" \
      $(geo_flags 8b) --threads $KVR_THREADS --qd $KVR_QD --mode $KVR_MODE --runs 1 --warmup 1
  done
fi

# C. SRPT coordinator with 2 processes: must finish one after the other, not together
if [ $ok = 1 ] && [ "$NGPU" -ge 2 ]; then
  python3 - "$SDIR/spec2.json" "$DATA_DIR" "${GPU_ARR[0]}" "${GPU_ARR[1]}" <<'EOF'
import json, sys
p, d, g0, g1 = sys.argv[1:]
json.dump([{"gpu": int(g0), "file": f"{d}/8b_0.bin", "geo": "8b"},
           {"gpu": int(g1), "file": f"{d}/8b_1.bin", "geo": "8b"}], open(p, "w"))
EOF
  for pol in fair srpt; do
    run python3 "$B/scripts/launch.py" --spec "$SDIR/spec2.json" --csv "$SDIR/coord2.csv" \
      --exp smoke --reps 2 --policy $pol --restore "$BIN/restore" --threads $KVR_THREADS \
      --qd $KVR_QD --mode $KVR_MODE --logdir "$SDIR/logs"
  done
  python3 - "$SDIR/coord2.csv" <<'EOF' | tee -a "$L"
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
good = True
for pol in ("fair", "srpt"):
    for rep in sorted({r["rep"] for r in rows}):
        e = sorted(float(r["end_rel_s"]) for r in rows
                   if r["policy"] == pol and r["rep"] == rep and r["end_rel_s"])
        if len(e) == 2:
            print(f"{pol} rep {rep}: end times {e[0]:.3f} {e[1]:.3f} ratio {e[1]/e[0]:.2f}")
            if pol == "srpt" and e[1] / e[0] <= 1.5:
                good = False
        elif pol == "srpt":
            good = False
print("SRPT_SERIALIZES", "yes" if good else "NO")
EOF
  grep -q "SRPT_SERIALIZES yes" "$L" || { echo "SRPT did not serialize the two restores" | tee -a "$L"; ok=0; }
fi

grep -E "mismatches|median|FAILED|VERIFY|SRPT_" "$L" | tail -40
[ $ok = 1 ] && stage_end 02_smoke ok || { stage_end 02_smoke fail "see smoke.log"; exit 1; }
