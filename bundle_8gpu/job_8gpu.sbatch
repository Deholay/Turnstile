#!/bin/bash
# Slurm template for the 8-GPU job. Fill in <ACCOUNT> and <PARTITION> (ask the human).
# Budget: 8 GPUs x 23 min = 3.07 GPU-hours (at 60 SU per GPU-hour: at most ~184 SU).
#SBATCH -A <ACCOUNT>
#SBATCH -p <PARTITION>
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH -c 96
#SBATCH -t 00:23:00
#SBATCH --signal=B:USR1@150
#SBATCH -o kvr_8gpu_%j.out
cd "$SLURM_SUBMIT_DIR"
TIME_LIMIT_MIN=23 exec bash run_8gpu.sh   # exec: the scheduler signal reaches run_8gpu.sh
