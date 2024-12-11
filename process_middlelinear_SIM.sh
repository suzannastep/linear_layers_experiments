#!/bin/bash

#SBATCH --job-name=process_middlelinear_SIM
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --output=log/process_middlelinear_SIM.out
#SBATCH --error=log/process_middlelinear_SIM.err
echo "$date Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"
echo "    CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
which python

python -W error::UserWarning process_results.py --Ls 2 3 4 5 6 7 8 9 --rs 1 --ns 64 128 256 512 1024 2048 --wds 1e-3 1e-4 1e-5 --labelnoise 0 0.25 --epochs 60100 --job_name middlelinear_SIM --architecture middlelinear --path /net/projects/willettlab/sueparkinson/teacher_networks/