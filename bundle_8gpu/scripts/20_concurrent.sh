#!/bin/bash
# Stage 20 (MAIN RESULT): G = 1, 2, 4, 8 GPUs restore one 32k history each at the same
# moment from the shared array. Policies: fair (independent, no coordination) and srpt
# (shared-memory grant to the restore with the fewest bytes left). Geometries 8b and 70b.
# Each process runs all REPS repetitions; launch.py starts every repetition on a common clock.
source "$(dirname "$0")/lib.sh"
stage_begin 20_concurrent
CSV="$SDIR/concurrent.csv"
start_sampler; trap stop_sampler EXIT
ok=1
Gs=(); for g in 1 2 4 8; do [ $g -le $NGPU ] && Gs+=($g); done
for geo in 8b 70b; do
  ensure_files $geo || { stage_end 20_concurrent fail "writing $geo files failed"; exit 1; }
  for G in "${Gs[@]}"; do
    python3 - "$SDIR/spec_${geo}_G$G.json" "$DATA_DIR" "$geo" "$G" "${GPU_ARR[@]}" <<'EOF'
import json, sys
p, d, geo, G, *gpus = sys.argv[1:]
json.dump([{"gpu": int(gpus[i]), "file": f"{d}/{geo}_{i}.bin", "geo": geo}
           for i in range(int(G))], open(p, "w"))
EOF
    for pol in fair srpt; do
      [ $G = 1 ] && [ $pol = srpt ] && continue   # identical to fair at G=1
      python3 "$B/scripts/launch.py" --spec "$SDIR/spec_${geo}_G$G.json" --csv "$CSV" \
        --exp concurrent_$geo --reps $REPS --policy $pol --restore "$BIN/restore" \
        --threads $KVR_THREADS --qd $KVR_QD --mode $KVR_MODE --logdir "$SDIR/logs" ||
        { echo "launch failed: $geo G=$G $pol" | tee -a "$SDIR/errors.txt"; ok=0; }
    done
    echo "done $geo G=$G $(date +%T)"
  done
done
[ $ok = 1 ] && stage_end 20_concurrent ok || stage_end 20_concurrent fail "see errors.txt"
