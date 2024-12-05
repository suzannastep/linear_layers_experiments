#!/bin/bash

#SBATCH --job-name=middlelinear_SIM_labelnoise0N64_L5_r1_wd1e-05_epochs60100
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --output=log/middlelinear_SIM_labelnoise0/N64_L5_r1_wd1e-05_epochs60100.out
#SBATCH --error=log/middlelinear_SIM_labelnoise0/N64_L5_r1_wd1e-05_epochs60100.err
echo "$date Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"
echo "    CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
which python

python -W error::UserWarning run_job.py --filename middlelinear_SIM_labelnoise0 --datasetsize 64 --L 5 --r 1 --labelnoise 0 --weight_decay 1e-05 --epochs 60100