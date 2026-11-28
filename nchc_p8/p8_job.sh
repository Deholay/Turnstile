#!/bin/bash
#SBATCH -A acd115218
#SBATCH -p normal2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH -t 00:08:00
#SBATCH -o /home/%u/p8/logs/p8_%j.out
# P8 (8 GPUs): the gaps left by the 11/20 8-GPU run, with the same restore code and settings
# (bundle_8gpu sources, copy mode, 8 threads, QD32, 5 repetitions, all pages verified).
#   A. load-vs-recompute crossover at G=8 for short histories (8B, 4k and 8k; G=1 and G=8, fair)
#   B. split without ordering: fair + recompute-front at G=8, same k/delays as the 11/20 SRPT run
#   C. under write-back: a fio writer at 10g (= 10.7 GB/s, as in Table 1 of the paper) runs while
#      C1 G=1 fair (single-restore bandwidth under write-back, input to the schedule)
#      C2 G=8 fair and srpt, load only
#      C3 G=8 srpt + recompute-front, k/delays planned for the write-back bandwidth
# Restore files live in node-local /tmp (md RAID0 of 8 NVMe) and are deleted at exit.
set -u
P=/home/$USER/p8
FIO=/work/$USER/killtest/envs/kt/bin/fio
# The compute nodes' system python3 is too old for launch.py (:= needs 3.8); use the kt2 env (3.12).
PY=/work/$USER/killtest/envs/kt2/bin/python
R=$P/results/p8-$(date +%F)-$SLURM_JOB_ID; mkdir -p $R/logs
D=$(mktemp -d /tmp/p8_${SLURM_JOB_ID}_XXXX)
SAMPLER=; WRITER=
cleanup() { [ -n "$WRITER" ] && kill $WRITER 2>/dev/null; [ -n "$SAMPLER" ] && kill $SAMPLER 2>/dev/null
            wait 2>/dev/null; rm -rf "$D"; }
trap cleanup EXIT
CMD=$R/commands.txt
GPUS=(0 1 2 3 4 5 6 7); G=8; REPS=5
L="$PY $P/launch.py --restore $P/restore --threads 8 --qd 32 --mode copy --reps $REPS --logdir $R/logs"
log() { echo "== $* $(date +%T)" | tee -a $CMD; }
{ echo "# job $SLURM_JOB_ID on $(hostname) $(date)"
  nvidia-smi --query-gpu=index,name,driver_version,pci.bus_id --format=csv,noheader
  cat /proc/mdstat | head -3; df -hT /tmp | tail -1; $FIO --version
  sha256sum $P/restore $P/src/* $P/launch.py $P/front_schedule.py $P/prefill.csv $P/p8_job.sh; } >> $CMD 2>&1
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { echo "GPUs busy, abort" | tee -a $CMD; exit 1; }

$PY $P/src/iotrace.py $R/diskstats.csv 0.05 & SAMPLER=$!

# --- restore files, one per GPU and geometry, written in parallel per geometry
flags() { case $1 in 70b) echo "--layers 80 --tokens 32768";; 8b) echo "--layers 32 --tokens 32768";;
                     8b4k) echo "--layers 32 --tokens 4096";; 8b8k) echo "--layers 32 --tokens 8192";; esac; }
log files
for geo in 70b 8b 8b4k 8b8k; do
  pids=()
  for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} $P/restore --file $D/${geo}_$i.bin --write $(flags $geo) \
      --kv-heads 8 --head-dim 128 >> $R/files_written.log 2>&1 & pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || { echo "writing $geo files failed" | tee -a $CMD; exit 1; }; done
done
log files done

# spec.json writer: spec GEO N [k list] [delay list]
spec() { $PY - "$@" <<'EOF'
import json, sys
out, geo, n, d, *rest = sys.argv[1:]
n = int(n); k = json.loads(rest[0]) if rest else [0] * n; dl = json.loads(rest[1]) if rest else [0.0] * n
json.dump([{"gpu": i, "file": f"{d}/{geo}_{i}.bin", "geo": geo, "start_tokens": k[i], "delay_s": dl[i]}
           for i in range(n)], open(out, "w"))
EOF
}
run() { log "$1 $3 G=$4"; $L --spec $2 --csv $R/$5.csv --exp $1 --policy $3 || echo "FAILED $1 $3" | tee -a $CMD; }

# --- A. crossover for short histories (recompute 4k/8k: 0.067/0.150 s, Table histlen)
for geo in 8b4k 8b8k; do
  spec $R/spec_${geo}_G1.json $geo 1 $D; run cross_$geo $R/spec_${geo}_G1.json fair 1 crossover
  spec $R/spec_${geo}_G8.json $geo 8 $D; run cross_$geo $R/spec_${geo}_G8.json fair 8 crossover
done

# --- B. fair + recompute-front with the 11/20 SRPT schedule (k_i, delay_i = t(k_i))
K8='[0,5504,8960,11776,13952,15872,17408,18816]'
T8='[0.0,0.092777,0.168208,0.236301,0.295076,0.348607,0.394631,0.438285]'
K70='[0,1792,3328,4736,6016,7168,8192,9216]'
T70='[0.0,0.231361,0.442745,0.65096,0.853076,1.047273,1.221257,1.399013]'
spec $R/spec_front_8b.json 8b 8 $D "$K8" "$T8";    run front_fair_8b $R/spec_front_8b.json fair 8 front_fair
spec $R/spec_front_70b.json 70b 8 $D "$K70" "$T70"; run front_fair_70b $R/spec_front_70b.json fair 8 front_fair

# --- C. under write-back
mkdir -p $D/wb
log writer start
$FIO --name=wb --filename=$D/wb/w.dat --size=32G --rw=write --bs=1M --iodepth=32 --direct=1 \
     --ioengine=libaio --rate=10g --time_based --runtime=900 --output-format=json \
     --output=$R/writer.json > /dev/null 2>&1 & WRITER=$!
sleep 15   # lay out the file and reach the rate
for geo in 8b 70b; do
  spec $R/spec_wb_${geo}_G1.json $geo 1 $D; run concurrent_$geo $R/spec_wb_${geo}_G1.json fair 1 wb_single
done
$PY $P/front_schedule.py --concurrent $R/wb_single.csv --prefill $P/prefill.csv --G $G \
  --out $R/schedule_wb.json | tee -a $CMD
for geo in 8b 70b; do
  spec $R/spec_wb_${geo}.json $geo 8 $D
  run wb_$geo $R/spec_wb_${geo}.json fair 8 writeback
  run wb_$geo $R/spec_wb_${geo}.json srpt 8 writeback
  $PY - $R/schedule_wb.json $geo $R/spec_wbfront_${geo}.json $D <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))[sys.argv[2]]
json.dump([{"gpu": i, "file": f"{sys.argv[4]}/{sys.argv[2]}_{i}.bin", "geo": sys.argv[2],
            "start_tokens": s["k"][i], "delay_s": s["delay_s"][i]} for i in range(s["G"])],
          open(sys.argv[3], "w"))
EOF
  run wbfront_$geo $R/spec_wbfront_${geo}.json srpt 8 writeback
done
kill $WRITER 2>/dev/null; wait $WRITER 2>/dev/null; WRITER=
log writer stopped

# --- summary: median over repetitions of mean and slowest response (end - delay for load only,
#     end for front runs, whose delay is the emulated recompute)
$PY - $R <<'EOF' | tee -a $CMD
import csv, glob, json, statistics as st, collections, sys, os
R = sys.argv[1]
for f in sorted(glob.glob(f"{R}/*.csv")):
    if f.endswith("diskstats.csv"): continue
    g = collections.defaultdict(list); mm = 0
    for r in csv.DictReader(open(f)):
        front = "front" in r["exp"]
        t = float(r["end_rel_s"]) - (0 if front else float(r["delay_s"]))
        g[(r["exp"], r["policy"], r["G"], r["rep"])].append(t); mm += int(r["mismatches"] or 0)
    agg = collections.defaultdict(lambda: ([], []))
    for (e, p, G, _), ts in g.items():
        agg[(e, p, G)][0].append(st.mean(ts)); agg[(e, p, G)][1].append(max(ts))
    print(f"# {os.path.basename(f)} mismatches={mm}")
    for k, (m, x) in sorted(agg.items()):
        print(f"RESULT {k[0]} {k[1]} G={k[2]} mean {st.median(m):.3f} max {st.median(x):.3f} reps {len(m)}")
try:
    w = json.load(open(f"{R}/writer.json"))["jobs"][0]["write"]
    print(f"WRITER {w['bw_bytes']/1e9:.2f} GB/s over {w['runtime']/1e3:.0f} s")
except Exception as e:
    print("WRITER summary failed:", e)
EOF
echo "# done $(date)" >> $CMD
