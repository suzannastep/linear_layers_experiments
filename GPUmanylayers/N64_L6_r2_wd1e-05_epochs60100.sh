#!/bin/bash

#SBATCH --job-name=GPUmanylayersN64_L6_r2_wd1e-05_epochs60100
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --output=log/GPUmanylayers/N64_L6_r2_wd1e-05_epochs60100.out
#SBATCH --error=log/GPUmanylayers/N64_L6_r2_wd1e-05_epochs60100.err
echo "$date Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"
echo "    CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
which python

python -W error::UserWarning run_job.py --filename GPUmanylayers --datasetsize 64 --L 6 --r 2 --weight_decay 1e-05 --epochs 60100