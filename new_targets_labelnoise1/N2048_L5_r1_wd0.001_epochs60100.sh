#!/bin/bash

#SBATCH --job-name=new_targets_labelnoise1N2048_L5_r1_wd0.001_epochs60100
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --output=log/new_targets_labelnoise1/N2048_L5_r1_wd0.001_epochs60100.out
#SBATCH --error=log/new_targets_labelnoise1/N2048_L5_r1_wd0.001_epochs60100.err
echo "$date Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"
echo "    CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
which python

python -W error::UserWarning run_job.py --filename new_targets_labelnoise1 --datasetsize 2048 --L 5 --r 1 --labelnoise 1 --weight_decay 0.001 --epochs 60100