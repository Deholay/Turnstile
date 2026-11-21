#!/bin/bash
#SBATCH -A acd115218
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH -t 00:25:00
#SBATCH -o /home/%u/p6/logs/p6_%j.out
# P6 (1 GPU): all single-GPU 32k TTFT numbers of tab:single inside ONE allocation, for 8B and
# 70B (the 11/15 numbers came from different allocations, which differ by ~10%).
# Per model, in priority order (each engine capped, a failure does not stop the rest):
#   ours       KvrSSDConnector load TTFT + recompute TTFT, same engine (ttft_kvr.py, kt2,
#              11/15 H200-tested harness; correctness checks off, done in jobs 472211/481677)
#   odirect    LMCache 0.3.5, chunk 256, use_odirect (its fastest disk setting)
#   dram       LMCache 0.3.5 serving from host memory (reference line)
#   default    LMCache 0.3.5, chunk 256, default disk backend
# Method as in P0/P2: 32,768 random history tokens + 128 new, TTFT = wall time of
# generate(max_tokens=1); LMCache files are fsynced and dropped from the page cache before
# every load. WARMUP=2, RUNS=3 (P0/P2 used 3+5) to fit the budget.
set -u
H=/home/$USER/harness
P=/home/$USER/p6
K=/work/$USER/killtest
KT2=$K/envs/kt2/bin/python    # vllm 0.10.1.1 (ours)
KT3=$K/envs/kt3/bin/python    # vllm 0.10.1.1 + lmcache 0.3.5 (P0 baselines)
M8=/home/$USER/models/Meta-Llama-3.1-8B-Instruct-FP8-dynamic
M70=$K/models/Llama-3.3-70B-Instruct-FP8-dynamic
export HF_HOME=/home/$USER/.cache/huggingface HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=$K/vllm_cache
export WARMUP=2 RUNS=3 CHECKS=0 PYTHONPATH=$H KVR_LIB=$H/libkvr.so
R=$P/results/p6-$(date +%F)-$SLURM_JOB_ID; mkdir -p $R
T=/tmp/p6_$SLURM_JOB_ID; mkdir -p $T
trap 'rm -rf $T' EXIT
CMD=$R/commands.txt
bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 | tr A-Z a-z)
N=$(cat /sys/bus/pci/devices/${bus#0000}/numa_node 2>/dev/null || cat /sys/bus/pci/devices/$bus/numa_node)
NB="numactl --cpunodebind=$N --membind=$N"; $NB true 2>/dev/null || NB=
{ echo "# job $SLURM_JOB_ID on $(hostname) $(date) gpu_numa=$N"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  $KT3 -c "import vllm,lmcache;print('kt3 vllm',vllm.__version__,'lmcache',getattr(lmcache,'__version__','?'))"
  sha256sum $H/kvr_core.cuh $H/kvr_lib.cu $H/kvr_connector.py $H/libkvr.so $P/*.py $P/p6_job.sh
  findmnt -no SOURCE,FSTYPE /tmp; } >> $CMD 2>&1

ours() { # tag model cap_s
  echo "== ours_$1 start $(date +%T)" | tee -a $CMD
  mkdir -p $T/kvr_$1
  KT_MODEL=$2 KVR_DIR=$T/kvr_$1 KVR_LOG=$R/kvr_$1.jsonl timeout $3 $NB $KT2 $P/ttft_kvr.py \
    $R/ours_$1.json > $R/ours_$1.log 2>&1
  echo "== ours_$1 exit $? $(date +%T)" | tee -a $CMD; grep MEDIAN $R/ours_$1.log | tee -a $CMD
  rm -rf $T/kvr_$1
}
lmc() { # tag model cfg cap_s ; cfg = default | odirect | dram
  local d=$T/lmc_$1_$3 y=$R/lmcache_$1_$3.yaml
  mkdir -p $d
  { echo "chunk_size: 256"
    [ $3 = dram ] && echo "local_cpu: true" || echo "local_cpu: false"
    echo "max_local_cpu_size: 24"; echo "max_local_disk_size: 60"; echo "local_disk: \"file://$d/\""
    [ $3 = odirect ] && printf 'extra_config:\n  use_odirect: true\n'; } > $y
  echo "== lmc_$1_$3 start $(date +%T)" | tee -a $CMD
  KT_MODEL=$2 KT_DISK_DIR=$d LMCACHE_CONFIG_FILE=$y timeout $4 $NB $KT3 $P/lmc_ttft.py lmcache \
    $R/lmc_$1_$3.json > $R/lmc_$1_$3.log 2>&1
  echo "== lmc_$1_$3 exit $? $(date +%T)" | tee -a $CMD; grep MEDIAN $R/lmc_$1_$3.log | tee -a $CMD
  rm -rf $d
}
ours 8b $M8 180
lmc 8b $M8 odirect 150
lmc 8b $M8 dram 150
lmc 8b $M8 default 150
ours 70b $M70 300
lmc 70b $M70 odirect 270
lmc 70b $M70 dram 240
lmc 70b $M70 default 270
echo "# done $(date)" >> $CMD
