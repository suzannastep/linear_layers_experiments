#!/bin/bash

#SBATCH --job-name=deeprelu_labelnoise0.25N64_L6_r1_wd0.0001_epochs60100
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --output=log/deeprelu_labelnoise0.25/N64_L6_r1_wd0.0001_epochs60100.out
#SBATCH --error=log/deeprelu_labelnoise0.25/N64_L6_r1_wd0.0001_epochs60100.err
echo "$date Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"
echo "    CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
which python

python -W error::UserWarning run_job.py --filename deeprelu_labelnoise0.25 --datasetsize 64 --L 6 --r 1 --labelnoise 0.25 --weight_decay 0.0001 --epochs 60100