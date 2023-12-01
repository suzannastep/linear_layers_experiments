#!/bin/bash

#SBATCH --job-name=many_layersN1024_L5_r1_wd0.0001_epochs60100
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --output=log/many_layers/N1024_L5_r1_wd0.0001_epochs60100.out
#SBATCH --error=log/many_layers/N1024_L5_r1_wd0.0001_epochs60100.err
echo "$date Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"
echo "    CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
conda activate cluster_startup
which python

python -W error::UserWarning run_job.py --filename many_layers --datasetsize 1024 --L 5 --r 1 --weight_decay 0.0001 --epochs 60100