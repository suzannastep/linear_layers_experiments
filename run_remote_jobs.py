import os
import numpy as np

# creates a folder in the current working directory
def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

# given input parameters, writes a .sh file and submits a sbatch job
def make_sbatch(code_file,out_file,datasetsize,L,r,labelnoise,weight_decay,epochs):
    paramname = f"N{datasetsize}_L{L}_r{r}_wd{weight_decay}_epochs{epochs}"
    job_file = f"{out_file}/{paramname}.sh"
    params  = "--filename {} --datasetsize {} --L {} --r {} --labelnoise {} --weight_decay {} --epochs {}"
    params = params.format(out_file,datasetsize,L,r,labelnoise,weight_decay,epochs)
    command = f"python -W error::UserWarning {code_file} "
    with open(job_file,'w') as fh:
        # the .sh file header may be different depending on the cluster
        fh.writelines('#!/bin/bash')
        fh.writelines('\n\n#SBATCH --job-name={}'.format(out_file+paramname))
        fh.writelines('\n#SBATCH --partition=general')
        fh.writelines('\n#SBATCH --gres=gpu:1')
        fh.writelines(f'\n#SBATCH --output=log/{out_file}/{paramname}.out')
        fh.writelines(f'\n#SBATCH --error=log/{out_file}/{paramname}.err')
        fh.writelines('\necho "$date Starting Job"')
        fh.writelines('\necho "SLURM Info: Job name:${SLURM_JOB_NAME}"')
        fh.writelines('\necho "    JOB ID: ${SLURM_JOB_ID}"')
        fh.writelines('\necho "    Host list: ${SLURM_JOB_NODELIST}"')
        fh.writelines('\necho "    CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"')
        fh.writelines('\nwhich python\n\n')
        fh.writelines(command+params)
    
    return job_file

def run_sbatch(job_file):
    os.system('sbatch {}'.format(job_file))
                                                                            
if __name__ == "__main__":
    code_file = '/home/sueparkinson/teacher_networks/run_job.py' #absolute path to file to run
    output_path = '/net/projects/willettlab/sueparkinson/teacher_networks' #absolute path to where to save results
    os.chdir(output_path)

    #parameters
    rs = [1]#[1,2,5]
    Ls = [2,3,4,5,6,7,8,9]
    wds = [1e-3,1e-4,1e-5]
    datasetsizes = [128,256,512]#[64,1024,2048]#[128,256,512]#,1024,2048]#[2048,1024,512,256,128,64]
    labelnoise = [0,0.25]#,0.5,1]
    epochs = 100_100
    jobname = "new_targets_SGD_more_epochs"

    #run files
    for r in rs:
        for ln in labelnoise:
            out_file = jobname + f"_labelnoise{ln}"
            #create folder for outputs
            mkdir(out_file)
            mkdir("log")
            mkdir(f"log/{out_file}")
            #run jobs
            for datasetsize in datasetsizes:
                for L in Ls:
                    for weight_decay in wds:
                        job_file = make_sbatch(code_file,out_file,datasetsize,L,r,ln,weight_decay,epochs)
                        print("created",job_file)
                        run_sbatch(job_file)
                        print("running",job_file)
