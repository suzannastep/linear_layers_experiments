#!/bin/bash

#SBATCH --job-name=many_layersN2048_L6_r2_wd1e-05_epochs60100
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --output=log/many_layers/N2048_L6_r2_wd1e-05_epochs60100.out
#SBATCH --error=log/many_layers/N2048_L6_r2_wd1e-05_epochs60100.err
echo "$date Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"
echo "    CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
conda activate cluster_startup
which python

python -W error::UserWarning run_job.py --filename many_layers --datasetsize 2048 --L 6 --r 2 --weight_decay 1e-05 --epochs 60100