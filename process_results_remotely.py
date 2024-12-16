import os
import numpy as np

def make_sbatch(Ls,rs,ns,wds,labelnoise,epochs,job_name,architecture,path,target):
    job_file = f"process_{job_name}"
    params  = "--Ls {} --rs {} --ns {} --wds {} --labelnoise {} --epochs {} --job_name {} --architecture {} --path {} --target {}"
    params = params.format(Ls,rs,ns,wds,labelnoise,epochs,job_name,architecture,path,target)
    command = f"python -W error::UserWarning process_results.py "
    with open(job_file + ".sh",'w') as fh:
        # the .sh file header may be different depending on the cluster
        fh.writelines('#!/bin/bash')
        fh.writelines('\n\n#SBATCH --job-name={}'.format(job_file))
        fh.writelines('\n#SBATCH --partition=general')
        fh.writelines('\n#SBATCH --gres=gpu:1')
        fh.writelines(f'\n#SBATCH --output=log/{job_file}.out')
        fh.writelines(f'\n#SBATCH --error=log/{job_file}.err')
        fh.writelines('\necho "$date Starting Job"')
        fh.writelines('\necho "SLURM Info: Job name:${SLURM_JOB_NAME}"')
        fh.writelines('\necho "    JOB ID: ${SLURM_JOB_ID}"')
        fh.writelines('\necho "    Host list: ${SLURM_JOB_NODELIST}"')
        fh.writelines('\necho "    CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"')
        fh.writelines('\nwhich python\n\n')
        fh.writelines(command+params)
    
    return job_file + ".sh"

def run_sbatch(job_file):
    os.system('sbatch {}'.format(job_file))

if __name__ == "__main__":
    path = "/net/projects/willettlab/sueparkinson/teacher_networks/"
    
    # #standard architecture
    # Ls = "2 3 4 5 6 7 8 9"
    # rs = "1 2 5"
    # ns = "64 128 256 512 1024 2048"
    # wds = "1e-3 1e-4 1e-5"
    # labelnoise = "0 0.25 0.5 1"
    # epochs = 60100
    # job_name = "new_targets"
    # architecture = "standard"
    # target = "SMIM"

    # job_file = make_sbatch(Ls,rs,ns,wds,labelnoise,epochs,job_name,architecture,path,target)
    # print("created",job_file)
    # run_sbatch(job_file)
    # print("running",job_file) 

    # #middlelinear
    # Ls = "2 3 4 5 6 7 8 9"
    # rs = "1"
    # ns = "64 128 256 512 1024 2048"
    # wds = "1e-3 1e-4 1e-5"
    # labelnoise = "0 0.25"
    # epochs = 60100
    # job_name = "middlelinear_SIM"
    # architecture = "middlelinear"
    # target = "SMIM"

    # job_file = make_sbatch(Ls,rs,ns,wds,labelnoise,epochs,job_name,architecture,path,target)
    # print("created",job_file)
    # run_sbatch(job_file)
    # print("running",job_file) 

    # #deeprelu
    # Ls = "2 3 4 5 6 7 8 9"
    # rs = "1"
    # ns = "64 128 256 512 1024 2048"
    # wds = "1e-3 1e-4 1e-5"
    # labelnoise = "0 0.25"
    # epochs = 60100
    # job_name = "deeprelu"
    # architecture = "relus"
    # target = "SMIM"

    # job_file = make_sbatch(Ls,rs,ns,wds,labelnoise,epochs,job_name,architecture,path,target)
    # print("created",job_file)
    # run_sbatch(job_file)
    # print("running",job_file) 

    # #SGD on standard architecture
    # Ls = "2 3 4 5 6 7 8 9"
    # rs = "1"
    # ns = "64 128 256 512 1024 2048"
    # wds = "1e-3 1e-4 1e-5"
    # labelnoise = "0 0.25"
    # epochs = 60100
    # job_name = "new_targets_SGD"
    # architecture = "standard"
    # target = "SMIM"

    # job_file = make_sbatch(Ls,rs,ns,wds,labelnoise,epochs,job_name,architecture,path,target)
    # print("created",job_file)
    # run_sbatch(job_file)
    # print("running",job_file) 

    #middlelinear specialized targets
    # Ls = "2 3 4 5 6 7 8 9"
    # rs = "1"
    # ns = "64 128 256 512 1024 2048"
    # wds = "1e-3 1e-4 1e-5"
    # labelnoise = "0 0.25"
    # epochs = 60100
    # job_name = "middlelinear_specializedtarget"
    # architecture = "middlelinear"
    # target = "specialized"

    # job_file = make_sbatch(Ls,rs,ns,wds,labelnoise,epochs,job_name,architecture,path,target)
    # print("created",job_file)
    # run_sbatch(job_file)
    # print("running",job_file) 

    #deeprelu specialized targets
    Ls = "2 3 4 5 6 7 8 9"
    rs = "1"
    ns = "64 128 256 512 1024 2048"
    wds = "1e-3 1e-4 1e-5"
    labelnoise = "0 0.25"
    epochs = 60100
    job_name = "deeprelu_specializedtarget"
    architecture = "relus"
    target = "specialized"

    job_file = make_sbatch(Ls,rs,ns,wds,labelnoise,epochs,job_name,architecture,path,target)
    print("created",job_file)
    run_sbatch(job_file)
    print("running",job_file) 
