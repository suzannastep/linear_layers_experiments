#!/bin/bash

#SBATCH --job-name=many_layers
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --output=log/many_layers.out
#SBATCH --error=log/many_layers.err
echo "$date Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"
echo "    CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
which python

python -W error::UserWarning run_job.py --filename many_layers --datasetsize 128 --L 2 --r 2 --weight_decay 1e-05 --epochs 60100