#!/bin/bash
#SBATCH -A acd115218
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH -t 00:15:00
#SBATCH -o /home/%u/p7_front/logs/p7_%j.out
# P7 (1 GPU, 8B): bundle stage 40 on the H200 — end-to-end recompute-front through vLLM 0.10.1.1.
# req1 = history[:k] (recompute, stays in the GPU prefix cache), req2 = full prompt, the
# connector loads only [k, N). Checks: load starts at k and covers only N-k tokens with matching
# checksums, and greedy output equals vLLM's prefix-cache path; times req2's overhead.
# Uses the current harness (suffix-only load), built in /home/$USER/p7_front with cuda/12.6;
# /home/$USER/harness (11/15 version) is not touched.
set -u
P=/home/$USER/p7_front
K=/work/$USER/killtest
KT2=$K/envs/kt2/bin/python
M8=/home/$USER/models/Meta-Llama-3.1-8B-Instruct-FP8-dynamic
export HF_HOME=/home/$USER/.cache/huggingface HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=$K/vllm_cache
R=$P/results/p7-$(date +%F)-$SLURM_JOB_ID; mkdir -p $R
D=/tmp/p7_$SLURM_JOB_ID; mkdir -p $D
trap 'rm -rf $D' EXIT
CMD=$R/commands.txt
bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 | tr A-Z a-z)
N=$(cat /sys/bus/pci/devices/${bus#0000}/numa_node 2>/dev/null || cat /sys/bus/pci/devices/$bus/numa_node)
NB="numactl --cpunodebind=$N --membind=$N"; $NB true 2>/dev/null || NB=
{ echo "# job $SLURM_JOB_ID on $(hostname) $(date) gpu_numa=$N"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  sha256sum $P/*.cuh $P/*.cu $P/*.h $P/*.py $P/restore $P/libkvr.so $P/p7_job.sh; } >> $CMD 2>&1

# A. fail fast: byte-exact partial restore with the new binary (8B geometry, 4k tokens, k=1024)
echo "== A_partial $(date +%T)" | tee -a $CMD
F=$D/small.bin
$P/restore --file $F --write --layers 32 --tokens 4096 > /dev/null
for s in 0 1024; do
  CUDA_VISIBLE_DEVICES=0 $NB $P/restore --file $F --layers 32 --tokens 4096 --start-tokens $s \
    --runs 1 --warmup 1 --mode copy | tail -1 | tee -a $CMD
  [ ${PIPESTATUS[0]} = 0 ] || { echo "partial restore check failed (start $s), abort" | tee -a $CMD; exit 1; }
done
rm -f $F

# B. stage 40 (same settings as bundle_8gpu/scripts/40_e2e_front.sh)
echo "== B_check_front $(date +%T)" | tee -a $CMD
mkdir -p $D/e2e
KT_MODEL=$M8 KVR_DIR=$D/e2e KVR_LIB=$P/libkvr.so KVR_LOG=$R/kvr_load.jsonl \
KF_HIST=32768 KF_KS="0,8192,16384,24576" KF_PROMPTS=2 PYTHONPATH=$P \
CUDA_VISIBLE_DEVICES=0 timeout 660 $NB $KT2 $P/check_front.py $R/check_front.json > $R/check_front.log 2>&1
echo "== B_check_front exit $? $(date +%T)" | tee -a $CMD
grep -E "^CHECK|^RESULT|Error|Traceback" $R/check_front.log | tail -20 | tee -a $CMD
echo "# done $(date)" >> $CMD
