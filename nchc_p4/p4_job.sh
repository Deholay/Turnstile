#!/bin/bash
#SBATCH -A acd115218
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH -t 00:30:00
#SBATCH -o /home/%u/p4/logs/p4_%j.out
# P4 (1 GPU): the single-GPU measurements still missing for the paper.
#   A. vLLM 0.32 native disk tier (OffloadingConnector, fs tier), 70B, 32k history + 128 new:
#      TTFT when the history is loaded from the SSD tier. Main single-GPU baseline.
#   B. 70B greedy check of our restore path against vLLM prefix caching (check_greedy.py),
#      on the H200-tested 11/15 harness (/home/$USER/harness, unchanged).
#   C. 8B recompute time at 32k/64k/96k/127k with chunked prefill (32k chunks); 4k..32k
#      unchunked are from job 472017, restore times for 4k..128k from job 481654.
# Second run (job 481654 fixes): A needed FlashInfer's precompiled kernels (installed with
# `vllm download-kernels`), B asserted on a None return value, C2 crashed with a 130k batch.
# Each part has its own timeout and writes its results as it goes; a failure in one part does
# not stop the others. A /proc/diskstats sampler runs for the whole job.
set -u
H=/home/$USER/harness          # 11/15 harness, H200-tested: restore, libkvr.so, kvr_connector.py
P=/home/$USER/p4               # this job's scripts
K=/work/$USER/killtest
KT2=$K/envs/kt2/bin/python     # vllm 0.10.1.1 + transformers 4.55.4
KT4=/home/$USER/envs/kt4/bin/python   # vllm 0.32.0+cu129
M8=/home/$USER/models/Meta-Llama-3.1-8B-Instruct-FP8-dynamic
M70=$K/models/Llama-3.3-70B-Instruct-FP8-dynamic
export HF_HOME=/home/$USER/.cache/huggingface HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=/home/$USER/.cache/vllm
R=$P/results/p4-$(date +%F)-$SLURM_JOB_ID; mkdir -p $R
D=/tmp/p4_$SLURM_JOB_ID; mkdir -p $D
CMD=$R/commands.txt
G=0
bus=$(nvidia-smi -i $G --query-gpu=pci.bus_id --format=csv,noheader | tr A-Z a-z)
N=$(cat /sys/bus/pci/devices/${bus#0000}/numa_node 2>/dev/null || cat /sys/bus/pci/devices/$bus/numa_node)
NB="numactl --cpunodebind=$N --membind=$N"; $NB true 2>/dev/null || NB=
{
  echo "# job $SLURM_JOB_ID on $(hostname) $(date) gpu_numa=$N"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,pci.bus_id --format=csv,noheader
  $KT2 -c "import vllm,torch,transformers;print('kt2 vllm',vllm.__version__,'torch',torch.__version__,'transformers',transformers.__version__)"
  $KT4 -c "import vllm,torch,transformers;print('kt4 vllm',vllm.__version__,'torch',torch.__version__,torch.version.cuda,'transformers',transformers.__version__)"
  sha256sum $H/kvr_core.cuh $H/kvr_lib.cu $H/restore.cu $H/kvr_connector.py $H/restore $H/libkvr.so \
            $P/*.py $P/p4_job.sh
  findmnt -no SOURCE,FSTYPE,OPTIONS /tmp; df -h /tmp | tail -1
} >> $CMD 2>&1

$KT2 $P/iotrace.py $R/diskstats.csv 0.05 & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null; rm -rf $D' EXIT
mark() { echo "$(date +%s.%N) $1" >> $R/phases.txt; echo "== $1 $(date +%T)" | tee -a $CMD; }
rc() { echo "== $1 exit $2 $(date +%T)" | tee -a $CMD; }

# A. vLLM 0.32 disk tier (12 min cap)
mark A_disk_tier
mkdir -p $D/fs
PATH=/home/$USER/envs/kt4/bin:$PATH CUDA_VISIBLE_DEVICES=$G timeout 720 $NB $KT4 $P/disk_tier_ttft.py --model $M70 --root $D/fs \
  --out $R/disk_tier_70b.json > $R/disk_tier_70b.log 2>&1
rc A_disk_tier $?
grep -E "^RESULT|Error|Traceback" $R/disk_tier_70b.log | tail -5 | tee -a $CMD
rm -rf $D/fs

# B. 70B greedy check, our restore path vs prefix caching (8 min cap)
mark B_greedy70
mkdir -p $D/kvr_70b
KT_MODEL=$M70 KVR_DIR=$D/kvr_70b KVR_LOG=$R/kvr_70b.jsonl KVR_LIB=$H/libkvr.so PYTHONPATH=$H \
  CUDA_VISIBLE_DEVICES=$G timeout 480 $NB $KT2 $P/check_greedy.py $R/greedy_70b.json > $R/greedy_70b.log 2>&1
rc B_greedy70 $?
grep -E "CHECK|RESULT|Error|Traceback" $R/greedy_70b.log | tail -8 | tee -a $CMD
rm -rf $D/kvr_70b

# C1 (restore time vs history length) completed in job 481654; not repeated.

# C2. 8B recompute time at 32k / 64k / 96k / 127k, chunked prefill (6 min cap)
mark C2_prefill_long
CUDA_VISIBLE_DEVICES=$G timeout 360 $NB $KT2 $P/prefill_vllm.py --model $M8 --tag 8b \
  --max-batched 32768 --lengths 32768 65536 98304 130048 --out $R/prefill_long_8b.csv > $R/prefill_long_8b.log 2>&1
rc C2_prefill_long $?
mark end
echo "# done $(date)" >> $CMD
