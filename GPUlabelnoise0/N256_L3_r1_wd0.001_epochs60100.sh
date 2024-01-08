#!/bin/bash

#SBATCH --job-name=GPUmanylayersN256_L3_r1_wd0.001_epochs60100
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --output=log/GPUmanylayers/N256_L3_r1_wd0.001_epochs60100.out
#SBATCH --error=log/GPUmanylayers/N256_L3_r1_wd0.001_epochs60100.err
echo "$date Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"
echo "    CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
which python

python -W error::UserWarning run_job.py --filename GPUmanylayers --datasetsize 256 --L 3 --r 1 --weight_decay 0.001 --epochs 60100