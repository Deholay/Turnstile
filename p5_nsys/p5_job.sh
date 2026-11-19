#!/bin/bash
#SBATCH -A acd115218
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH -c 12
#SBATCH -t 00:05:00
#SBATCH -o /home/%u/p5_nsys/logs/p5_%j.out
# P5 (1 GPU): nsys timeline of the default restore config (copy, 11 threads, QD32, 70B geometry,
# 10 GiB) for the overlap figure. ./restore = current harness + NVTX ranges (-DKVR_NVTX).
#   A. ./restore without nsys            -> time of this build
#   B. 11/15 harness binary, no nsys     -> same allocation, reference for Table ablation
#   C. ./restore under nsys              -> main trace (must stay close to A)
#   D. ./restore --no-overlap under nsys -> contrast panel
set -u
P=/home/$USER/p5_nsys; H=/home/$USER/harness
R=$P/results/$SLURM_JOB_ID; mkdir -p $R
D=/tmp/kvtest_$SLURM_JOB_ID; mkdir -p $D
trap 'rm -rf $D' EXIT
CSV=$R/restore.csv; CMD=$R/commands.txt
module load cuda/12.6 >/dev/null 2>&1
NSYS=$(command -v nsys || echo /usr/local/cuda/bin/nsys)
GPUBUS=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 | tr A-Z a-z)
GPUNUMA=$(cat /sys/bus/pci/devices/${GPUBUS#0000}/numa_node 2>/dev/null || cat /sys/bus/pci/devices/$GPUBUS/numa_node)
NB="numactl --cpunodebind=$GPUNUMA --membind=$GPUNUMA"; $NB true || NB=""
{
  echo "# job $SLURM_JOB_ID on $(hostname) $(date) gpu_numa=$GPUNUMA nsys=$NSYS"
  nvidia-smi --query-gpu=name,driver_version,pci.bus_id --format=csv,noheader
  $NSYS --version
  sha256sum $P/restore.cu $P/kvr_core.cuh $P/restore $H/restore $P/p5_job.sh
} >> $CMD 2>&1
A="--file $D/kv512k.bin --mode copy --threads 11 --qd 32"
$NB $P/restore --file $D/kv512k.bin --write | tee -a $CMD
echo "$NB $P/restore $A --label p5_copy_t11_q32" >> $CMD
$NB $P/restore $A --csv $CSV --label p5_copy_t11_q32 2>&1 | tail -2
echo "$NB $H/restore $A --label h1115_copy_t11_q32" >> $CMD
$NB $H/restore $A --csv $CSV --label h1115_copy_t11_q32 2>&1 | tail -2
NS="$NSYS profile -t cuda,nvtx -s none --cpuctxsw=none --force-overwrite true"
echo "$NB $NS -o $R/copy_t11_q32 $P/restore $A --runs 1 --warmup 1" >> $CMD
$NB $NS -o $R/copy_t11_q32 $P/restore $A --runs 1 --warmup 1 --csv $CSV --label nsys_copy_t11_q32 2>&1 | grep -v '^\[1/1\]' | tail -6
echo "$NB $NS -o $R/nooverlap_copy_t11_q32 $P/restore $A --no-overlap --runs 1 --warmup 1" >> $CMD
$NB $NS -o $R/nooverlap_copy_t11_q32 $P/restore $A --no-overlap --runs 1 --warmup 1 --csv $CSV --label nsys_nooverlap_copy_t11_q32 2>&1 | grep -v '^\[1/1\]' | tail -6
echo "# done $(date)" >> $CMD
