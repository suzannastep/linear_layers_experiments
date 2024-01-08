#!/bin/bash

#SBATCH --job-name=GPUlabelnoise0.25N2048_L6_r2_wd0.001_epochs60100
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --output=log/GPUlabelnoise0.25/N2048_L6_r2_wd0.001_epochs60100.out
#SBATCH --error=log/GPUlabelnoise0.25/N2048_L6_r2_wd0.001_epochs60100.err
echo "$date Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"
echo "    CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
which python

python -W error::UserWarning run_job.py --filename GPUlabelnoise0.25 --datasetsize 2048 --L 6 --r 2 --labelnoise 0.25 --weight_decay 0.001 --epochs 60100