#!/bin/bash
# Slurm template for the 1-GPU vLLM job. Fill in <ACCOUNT> and <PARTITION> (ask the human).
# Budget: 1 GPU x 12 min = 0.2 GPU-hours (at 60 SU per GPU-hour: at most ~12 SU).
#SBATCH -A <ACCOUNT>
#SBATCH -p <PARTITION>
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH -t 00:12:00
#SBATCH -o kvr_1gpu_%j.out
cd "$SLURM_SUBMIT_DIR"
bash run_1gpu.sh
