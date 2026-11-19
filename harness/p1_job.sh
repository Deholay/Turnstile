#!/bin/bash
#SBATCH -A acd115218
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH -c 12
#SBATCH -t 00:15:00
#SBATCH -o /home/%u/harness/logs/p1_%j.out
# P1: restore harness on H200 + node-local RAID0 (/tmp). Every run appends to CSV immediately,
# so a timeout keeps everything measured so far. Ordered by importance.
set -u
H=/home/$USER/harness
R=$H/results/p1-$(date +%F)-$SLURM_JOB_ID; mkdir -p $R
D=/tmp/kvtest_$SLURM_JOB_ID; mkdir -p $D
CSV=$R/restore.csv; CMD=$R/commands.txt
trap 'rm -rf $D' EXIT
module load cuda/12.6 >/dev/null 2>&1
GPUBUS=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 | tr A-Z a-z)
GPUNUMA=$(cat /sys/bus/pci/devices/${GPUBUS#0000}/numa_node 2>/dev/null || cat /sys/bus/pci/devices/$GPUBUS/numa_node)
OTHER=$((1 - GPUNUMA))
NB="numactl --cpunodebind=$GPUNUMA --membind=$GPUNUMA"; $NB true || NB=""
{
  echo "# job $SLURM_JOB_ID on $(hostname) $(date) gpu_numa=$GPUNUMA"
  nvidia-smi --query-gpu=name,driver_version,pci.bus_id --format=csv,noheader
  sha256sum $H/restore.cu $H/p1_job.sh $H/restore
  df -h /tmp | tail -1
} >> $CMD 2>&1

run() { # label prefix-cmd args...
  local label=$1 pre=$2; shift 2
  echo "$pre $H/restore $* --csv $CSV --label $label" >> $CMD
  $pre $H/restore "$@" --csv $CSV --label $label 2>&1 | tail -2
  echo "== $label exit ${PIPESTATUS[0]} $(date +%T)"
}

# A. correctness on a small geometry, both modes
$H/restore --file $D/small.bin --write --layers 4 --tokens 2048
run small_direct "$NB" --file $D/small.bin --layers 4 --tokens 2048 --runs 2 --warmup 1
run small_copy   "$NB" --file $D/small.bin --layers 4 --tokens 2048 --runs 2 --warmup 1 --mode copy
# stop early (saves SU) if the small check did not pass in both modes
[ "$(grep -c '^small_' $CSV 2>/dev/null)" = 4 ] || { echo "small check failed, abort" | tee -a $CMD; exit 1; }

# B. 70B geometry, 32k tokens, 10 GiB, 512 KiB items
F=$D/kv512k.bin
$NB $H/restore --file $F --write | tee -a $CMD

# C. headline + thread/QD sweep (direct kernel)
run main_t8_q16 "$NB" --file $F --threads 8 --qd 16 --mode direct
for t in 1 2 4 8 11; do for q in 4 16 32; do
  run direct_t${t}_q${q} "$NB" --file $F --threads $t --qd $q --mode direct
done; done
run direct_t1_q1 "$NB" --file $F --threads 1 --qd 1 --mode direct

# D. copy engine + scatter
for t in 4 8 11; do for q in 16 32; do
  run copy_t${t}_q${q} "$NB" --file $F --threads $t --qd $q --mode copy
done; done

# E. no overlap (read everything, then scatter) and LMCache-like (1 thread, QD1, no overlap)
run nooverlap_t8_q16 "$NB" --file $F --threads 8 --qd 16 --no-overlap --mode direct
run lmcachelike_t1_q1 "$NB" --file $F --threads 1 --qd 1 --no-overlap --mode direct

# F. NUMA: no binding, opposite node
run numa_none_t8_q16 "" --file $F --threads 8 --qd 16 --mode direct
run numa_other_t8_q16 "numactl --cpunodebind=$OTHER --membind=$OTHER" --file $F --threads 8 --qd 16 --mode direct

# G. ring / batch
for r in 64 128 512; do run ring${r}_t8_q16 "$NB" --file $F --threads 8 --qd 16 --ring $r; done
for b in 4 16 64; do run batch${b}_t8_q16 "$NB" --file $F --threads 8 --qd 16 --batch $b; done

# H. item size 256 KiB / 1 MiB (same bytes in flight)
$NB $H/restore --file $D/kv256k.bin --write --seg-tokens 64 >> $CMD
run seg256k_t8_q32 "$NB" --file $D/kv256k.bin --seg-tokens 64 --threads 8 --qd 32 --ring 512 --mode direct
rm -f $D/kv256k.bin
$NB $H/restore --file $D/kv1m.bin --write --seg-tokens 256 >> $CMD
run seg1m_t8_q8 "$NB" --file $D/kv1m.bin --seg-tokens 256 --threads 8 --qd 8 --ring 128 --mode direct
rm -f $D/kv1m.bin

# I. one nsys timeline of the headline config (if available)
if command -v nsys >/dev/null; then
  $NB nsys profile -t cuda,osrt -o $R/main_t8_q16 --force-overwrite true \
    $H/restore --file $F --threads 8 --qd 16 --runs 1 --warmup 1 2>&1 | tail -3
fi
echo "# done $(date)" >> $CMD
