#!/bin/bash
#SBATCH -A acd115218
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH -c 12
#SBATCH -t 00:20:00
#SBATCH -o /home/%u/harness/logs/p2_%j.out
# P2: KvrSSDConnector in vLLM 0.10.1.1 — 32k TTFT (load vs recompute) and correctness.
# 8B first (fast, fails early), then 70B. KV files on node-local RAID0 /tmp.
set -u
H=/home/$USER/harness
K=/work/$USER/killtest
export PATH=$K/envs/kt2/bin:$PATH HF_HOME=$K/hf_home VLLM_CACHE_ROOT=$K/vllm_cache
export PYTHONPATH=$H KVR_LIB=$H/libkvr.so
R=$H/results/p2-$(date +%F)-$SLURM_JOB_ID; mkdir -p $R
CMD=$R/commands.txt
GPUBUS=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 | tr A-Z a-z)
GPUNUMA=$(cat /sys/bus/pci/devices/${GPUBUS#0000}/numa_node 2>/dev/null || cat /sys/bus/pci/devices/$GPUBUS/numa_node)
NB="numactl --cpunodebind=$GPUNUMA --membind=$GPUNUMA"; $NB true || NB=""
{
  echo "# job $SLURM_JOB_ID on $(hostname) $(date) gpu_numa=$GPUNUMA"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  python -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__)"
  sha256sum $H/kvr_core.cuh $H/kvr_lib.cu $H/kvr_connector.py $H/ttft_kvr.py $H/p2_job.sh $H/libkvr.so
} >> $CMD 2>&1

# refactored pipeline: small byte-exact check in both modes before loading any model
S=/tmp/kvr_${SLURM_JOB_ID}_small.bin
$H/restore --file $S --write --layers 4 --tokens 2048 > /dev/null
for m in copy direct; do
  $NB $H/restore --file $S --layers 4 --tokens 2048 --runs 1 --warmup 1 --mode $m | tail -1 | tee -a $CMD
  [ ${PIPESTATUS[0]} = 0 ] || { echo "harness check failed ($m), abort" | tee -a $CMD; exit 1; }
done
rm -f $S

run() { # tag model
  export KT_MODEL=$2 KVR_DIR=/tmp/kvr_${SLURM_JOB_ID}_$1 KVR_LOG=$R/kvr_$1.jsonl
  mkdir -p $KVR_DIR
  echo "$NB python $H/ttft_kvr.py $R/ttft_$1.json  # KT_MODEL=$2" >> $CMD
  $NB python $H/ttft_kvr.py $R/ttft_$1.json 2>&1 | grep -E "ROW|CHECK|MEDIAN|kvr|Error|error|Traceback|assert" | grep -v "it/s]"
  echo "== $1 exit ${PIPESTATUS[0]} $(date +%T)" | tee -a $CMD
  rm -rf $KVR_DIR
}
run 8b /home/$USER/models/Meta-Llama-3.1-8B-Instruct-FP8-dynamic
grep -q '"greedy_equal"' $R/ttft_8b.json 2>/dev/null || { echo "8b failed, skip 70b" | tee -a $CMD; exit 1; }
run 70b $K/models/Llama-3.3-70B-Instruct-FP8-dynamic
echo "# done $(date)" >> $CMD
