#!/bin/bash
#SBATCH -A acd115218
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH -c 12
#SBATCH -t 00:05:00
#SBATCH -o /home/%u/p5_nsys/logs/p5b_%j.out
# P5b (1 GPU): is the 50 ms read stall in the t11 nsys trace a profiler artifact?
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
  sha256sum $P/restore.cu $P/kvr_core.cuh $P/restore $H/restore $P/p5b_job.sh
} >> $CMD 2>&1
$NB $P/restore --file $D/kv512k.bin --write | tee -a $CMD
B="--file $D/kv512k.bin --mode copy --qd 32"
#   A. t11 untraced, built-in I/O trace (CLOCK_MONOTONIC submit/complete per read)
$NB $P/restore $B --threads 11 --runs 3 --warmup 2 --trace $R/io_t11.csv --csv $CSV --label iotrace_copy_t11_q32 2>&1 | tail -1
#   B. t8 untraced reference, C. t8 under nsys (leaves 3 of 12 cores free for the profiler)
$NB $P/restore $B --threads 8 --csv $CSV --label p5_copy_t8_q32 2>&1 | tail -1
NS="$NSYS profile -t cuda,nvtx -s none --cpuctxsw=none --force-overwrite true"
echo "$NB $NS -o $R/copy_t8_q32 $P/restore $B --threads 8 --runs 1 --warmup 1" >> $CMD
$NB $NS -o $R/copy_t8_q32 $P/restore $B --threads 8 --runs 1 --warmup 1 --csv $CSV --label nsys_copy_t8_q32 2>&1 | grep -v '^\[1/1\]' | tail -3
echo "# done $(date)" >> $CMD
