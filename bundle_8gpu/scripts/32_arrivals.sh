#!/bin/bash
# Stage 32: restores that do not arrive together. G restores (one per GPU) arrive as a Poisson
# process; mean inter-arrival time = x * (single-restore time), x in {0.25, 0.5, 1}. fair vs
# srpt, load only. Arrival times are seeded per repetition and identical for both policies.
# Response time of restore i = end_rel_s - delay_s (from its own arrival).
source "$(dirname "$0")/lib.sh"
stage_begin 32_arrivals
G=$NGPU; [ $G -gt 8 ] && G=8
C="$OUT_DIR/20_concurrent/concurrent.csv"
[ -f "$C" ] || { stage_end 32_arrivals skip "stage 20 results missing"; exit 0; }
CSV="$SDIR/arrivals.csv"
start_sampler; trap stop_sampler EXIT
ok=1
for geo in 8b 70b; do
  for x in 0.25 0.5 1; do
    python3 - "$C" "$SDIR/spec.json" "$DATA_DIR" "$geo" "$G" "$x" "$REPS" "${GPU_ARR[@]}" <<'EOF'
import csv, json, random, statistics, sys
c, p, d, geo, G, x, reps, *gpus = sys.argv[1:]
G, x, reps = int(G), float(x), int(reps)
single = statistics.median(float(row["restore_s"]) for row in csv.DictReader(open(c))
                           if row["exp"] == f"concurrent_{geo}" and row["G"] == "1" and row["restore_s"])
delays = [[0.0] * reps for _ in range(G)]
for r in range(reps):
    rng = random.Random(1000 * r + int(x * 100))
    t = 0.0
    order = list(range(G))
    rng.shuffle(order)                      # which GPU arrives k-th varies per repetition
    for i in order:
        delays[i][r] = t
        t += rng.expovariate(1.0 / (x * single))
json.dump([{"gpu": int(gpus[i]), "file": f"{d}/{geo}_{i}.bin", "geo": geo, "delay_s": delays[i]}
           for i in range(G)], open(p, "w"))
EOF
    for pol in fair srpt; do
      python3 "$B/scripts/launch.py" --spec "$SDIR/spec.json" --csv "$CSV" \
        --exp arrivals_${geo}_x$x --reps $REPS --policy $pol --restore "$BIN/restore" \
        --threads $KVR_THREADS --qd $KVR_QD --mode $KVR_MODE ||
        { echo "launch failed: $geo x=$x $pol" | tee -a "$SDIR/errors.txt"; ok=0; }
    done
  done
done
[ $ok = 1 ] && stage_end 32_arrivals ok || stage_end 32_arrivals fail "see errors.txt"
