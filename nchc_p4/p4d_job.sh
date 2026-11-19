#!/bin/bash
#SBATCH -A acd115218
#SBATCH -p normal2
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH -t 00:15:00
#SBATCH -o /home/%u/p4/logs/p4d_%j.out
# P4-debug (1 GPU): job 481677's vLLM 0.32 disk-tier run segfaulted in the engine right after
# the first 70B store, with no stack trace, and the main process then hung until the timeout.
# Here the engine runs in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0), so a segfault ends the
# run at once and faulthandler prints the Python stack of every thread. Three short runs
# (2 prompts each: 1 warmup + 1 timed), each capped at 4 minutes:
#   1. 8B,  O_DIRECT     is the crash model-specific?
#   2. 70B, O_DIRECT     reproduce with a stack trace
#   3. 70B, page cache   is the O_DIRECT write path the cause? (loads may hit DRAM: not a
#                        disk-tier number, diagnosis only)
set -u
P=/home/$USER/p4
K=/work/$USER/killtest
KT4=/home/$USER/envs/kt4/bin/python
M8=/home/$USER/models/Meta-Llama-3.1-8B-Instruct-FP8-dynamic
M70=$K/models/Llama-3.3-70B-Instruct-FP8-dynamic
export HF_HOME=/home/$USER/.cache/huggingface HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=/home/$USER/.cache/vllm
export PATH=/home/$USER/envs/kt4/bin:$PATH PYTHONFAULTHANDLER=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
R=$P/results/p4d-$(date +%F)-$SLURM_JOB_ID; mkdir -p $R
D=/tmp/p4d_$SLURM_JOB_ID; mkdir -p $D
trap 'rm -rf $D' EXIT
CMD=$R/commands.txt
{ echo "# job $SLURM_JOB_ID on $(hostname) $(date)"; ulimit -a | grep -i "core\|stack\|locked"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; sha256sum $P/disk_tier_ttft.py $P/p4d_job.sh
  findmnt -no SOURCE,FSTYPE,OPTIONS /tmp; } >> $CMD 2>&1
run() { # tag model extra-args
  local tag=$1 m=$2; shift 2
  echo "== $tag start $(date +%T)" | tee -a $CMD
  rm -rf $D/fs; mkdir -p $D/fs
  CUDA_VISIBLE_DEVICES=0 timeout 240 $KT4 $P/disk_tier_ttft.py --model $m --root $D/fs \
    --out $R/$tag.json --warmup 1 --runs 1 --settle 5 --disk-gib 24 "$@" > $R/$tag.log 2>&1
  local e=$?
  echo "== $tag exit $e $(date +%T)" | tee -a $CMD
  grep -E "^(engine|store|load|RESULT)|Segfault|Fatal Python|Error" $R/$tag.log | head -12 | tee -a $CMD
}
run d1_8b_odirect $M8
run d2_70b_odirect $M70
run d3_70b_pagecache $M70 --page-cache
echo "# done $(date)" >> $CMD
