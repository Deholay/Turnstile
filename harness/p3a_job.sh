#!/bin/bash
#SBATCH -A acd115218
#SBATCH -p normal2
#SBATCH --gres=gpu:2
#SBATCH --gres-flags=enforce-binding
#SBATCH -c 24
#SBATCH -t 00:15:00
#SBATCH -o /home/%u/harness/logs/p3a_%j.out
# P3a (first P3 dev job, 2 GPUs). Everything here runs on code that already exists:
#   A. small byte-exact check, both modes (abort on failure)
#   B. I/O traces for MSS + uncoordinated shared-SSD baseline at G=2 (restore harness, 70B
#      geometry): GPU0 alone, then GPU0 and GPU1 together on their own files, same start.
#   C. the supplementary greedy check (vs vLLM prefix caching): 8B on GPU0 and 70B on GPU1
#      in parallel, with per-read traces of every kvr load.
# A /proc/diskstats sampler (iotrace.py, 50 ms) runs for the whole job.
set -u
H=/home/$USER/harness
K=/work/$USER/killtest
export PATH=$K/envs/kt2/bin:$PATH HF_HOME=$K/hf_home VLLM_CACHE_ROOT=$K/vllm_cache
export PYTHONPATH=$H KVR_LIB=$H/libkvr.so
R=$H/results/p3a-$(date +%F)-$SLURM_JOB_ID; mkdir -p $R
D=/tmp/p3a_$SLURM_JOB_ID; mkdir -p $D
CMD=$R/commands.txt
numa_of() { # physical GPU index -> NUMA node
  local bus=$(nvidia-smi -i $1 --query-gpu=pci.bus_id --format=csv,noheader | tr A-Z a-z)
  cat /sys/bus/pci/devices/${bus#0000}/numa_node 2>/dev/null || cat /sys/bus/pci/devices/$bus/numa_node
}
G0=0; G1=1
N0=$(numa_of $G0); N1=$(numa_of $G1)
nb() { local c="numactl --cpunodebind=$1 --membind=$1"; $c true 2>/dev/null && echo "$c"; }
NB0=$(nb $N0); NB1=$(nb $N1)
{
  echo "# job $SLURM_JOB_ID on $(hostname) $(date) gpu_numa=$N0,$N1"
  nvidia-smi --query-gpu=index,name,driver_version,pci.bus_id --format=csv,noheader
  python -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__)"
  sha256sum $H/kvr_core.cuh $H/kvr_lib.cu $H/restore.cu $H/kvr_connector.py $H/check_greedy.py \
            $H/iotrace.py $H/p3a_job.sh $H/restore $H/libkvr.so
  findmnt -no SOURCE,FSTYPE,OPTIONS /tmp; cat /proc/mdstat 2>/dev/null; df -h /tmp | tail -1
} >> $CMD 2>&1

python $H/iotrace.py $R/diskstats.csv 0.05 & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null; rm -rf $D' EXIT
mark() { echo "$(python -c 'import time;print(f"{time.monotonic():.6f}")') $1" >> $R/phases.txt; }

# A. small check
mark A_small
$H/restore --file $D/small.bin --write --layers 4 --tokens 2048 > /dev/null
for m in copy direct; do
  CUDA_VISIBLE_DEVICES=$G0 $NB0 $H/restore --file $D/small.bin --layers 4 --tokens 2048 \
    --runs 1 --warmup 1 --mode $m | tail -1 | tee -a $CMD
  [ ${PIPESTATUS[0]} = 0 ] || { echo "harness check failed ($m), abort" | tee -a $CMD; exit 1; }
done

# B. traces + G=2 uncoordinated baseline (copy mode, t11 q32 = P2 setting)
mark B_write
CUDA_VISIBLE_DEVICES=$G0 $NB0 $H/restore --file $D/f0.bin --write >> $CMD
CUDA_VISIBLE_DEVICES=$G1 $NB1 $H/restore --file $D/f1.bin --write >> $CMD
RA="--threads 11 --qd 32 --mode copy --warmup 1 --runs 5"
out() { echo "--csv $R/restore_$1.csv --trace $R/io_restore_$1.csv --label $1"; }  # one file per process
mark B_alone
echo "restore --file f0 $RA --label alone_g0" >> $CMD
CUDA_VISIBLE_DEVICES=$G0 $NB0 $H/restore --file $D/f0.bin $RA $(out alone_g0) | tail -1 | tee -a $CMD
mark B_conc2
GO=$D/go; rm -f $GO
echo "restore --file f{0,1} $RA --go-file go --label conc2_g{0,1}  (started together)" >> $CMD
CUDA_VISIBLE_DEVICES=$G0 $NB0 $H/restore --file $D/f0.bin $RA $(out conc2_g0) --go-file $GO > $R/conc2_g0.txt 2>&1 & P0=$!
CUDA_VISIBLE_DEVICES=$G1 $NB1 $H/restore --file $D/f1.bin $RA $(out conc2_g1) --go-file $GO > $R/conc2_g1.txt 2>&1 & P1=$!
sleep 5; touch $GO
wait $P0; E0=$?; wait $P1; E1=$?
tail -1 $R/conc2_g0.txt $R/conc2_g1.txt | tee -a $CMD
echo "== conc2 exit $E0 $E1 $(date +%T)" | tee -a $CMD
rm -f $D/f0.bin $D/f1.bin

# C. greedy check vs prefix caching, 8B || 70B
mark C_greedy
greedy() { # tag gpu numa-prefix model
  KT_MODEL=$4 KVR_DIR=$D/kvr_$1 KVR_LOG=$R/kvr_$1.jsonl KVR_TRACE=$R/io_load_$1.csv \
    CUDA_VISIBLE_DEVICES=$2 $3 python $H/check_greedy.py $R/greedy_$1.json > $R/greedy_$1.log 2>&1
  local e=$?
  grep -E "CHECK|RESULT|Error|Traceback" $R/greedy_$1.log | tee -a $CMD
  echo "== greedy $1 exit $e $(date +%T)" | tee -a $CMD
}
mkdir -p $D/kvr_8b $D/kvr_70b
echo "check_greedy.py 8b on GPU$G0, 70b on GPU$G1 in parallel" >> $CMD
greedy 8b $G0 "$NB0" /home/$USER/models/Meta-Llama-3.1-8B-Instruct-FP8-dynamic &
greedy 70b $G1 "$NB1" $K/models/Llama-3.3-70B-Instruct-FP8-dynamic &
wait
mark end
echo "# done $(date)" >> $CMD
