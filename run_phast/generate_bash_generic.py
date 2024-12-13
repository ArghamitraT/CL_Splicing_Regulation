

import os
import time
import random

trimester = time.strftime("_%Y_%m_%d__%H_%M_%S")
def create_job_dir(dir="", fold_name = ""):
    if dir:
        job_path = os.path.join(dir, fold_name)
        os.mkdir(job_path)

    else:
        job_path = os.path.join(os.getcwd(), fold_name)
        if not os.path.exists(job_path):
            os.mkdir(job_path)

    return job_path

def create_prg_file(prg_file_path, python_script):
   
    header = f"#!/bin/bash\n" + \
    "set -e\n" + \
    "cd $HOME\n" + \
    "source ~/.bashrc\n" + \
    f"conda activate {environment}\n" + \
    "module load phast/1.5\n" + \
    f"python {python_script}"
 

    with open(prg_file_path, "w") as f:
        f.write(header)
    return prg_file_path
    
    
def copy_weights(desired_weight_path, to_be_saved_path ):
    for file_name in os.listdir(desired_weight_path):
        dir = os.path.join(desired_weight_path, file_name)
        os.system(f"cp {dir} {to_be_saved_path}")


def create_slurm_file(prg_file_path, job_name, slurm_file_path):

    header = f"#!/bin/bash\n" + \
    "##ENVIRONMENT SETTINGS; REPLACE WITH CAUTION\n" + \
    "##NECESSARY JOB SPECIFICATIONS\n" + \
    f"#SBATCH --job-name={job_name}      #Set the job name to \"JobExample1\"\n" + \
    f"#SBATCH --time={hour}:45:00              #Set the wall clock limit to 1hr and 30min,\n" + \
    f"#SBATCH --mem={memory}G              \n" + \
    f"#SBATCH --cpus-per-task={cpu}                   \n" + \
    "#SBATCH --mail-type=END,FAIL    \n" + \
    f"#SBATCH --output={output_dir}/out_{job_name}.%j      #Send stdout/err to\n" + \
    "#SBATCH --mail-user=atalukder@nygenome.org                    \n" + \
    f"{prg_file_path}"

    with open (slurm_file_path, "w") as f:
        f.write(header)
    return slurm_file_path


def get_file_name(kind, l0=0, l1=0, l2=0, l3=0, ext=True):

    file_name = f"{kind}_{trimester}"
    if ext:
        file_name = f"{file_name}.sh"
    return file_name



main_data_dir = "/gpfs/commons/home/atalukder/Contrastive_Learning/files/results"
job_path = "/gpfs/commons/home/atalukder/Contrastive_Learning/files/cluster_job_submission_files"
code_dir = "/gpfs/commons/home/atalukder/Contrastive_Learning/code"
where_to_save = "/gpfs/commons/home/atalukder/Contrastive_Learning/files/results/saved_weights_toResume"

data_dir_0   = create_job_dir(dir= main_data_dir, fold_name= "exprmnt"+trimester)
data_dir   = create_job_dir(dir= data_dir_0, fold_name= "files")
weight_dir = create_job_dir(dir= data_dir_0, fold_name="weights")
output_dir = create_job_dir(dir= data_dir, fold_name="output_files")


""" Parameters: **CHANGE (AT)** """
input_data_dir = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/"
code_dir = '/gpfs/commons/home/atalukder/Contrastive_Learning/code/run_phast'
main_script = 'convert_wigToPkl.py'
scripts_tobe_saved= ['generate_bash_generic.py']
name = "run_phast"
hour = 20 # hour
memory = 80 #GB
cpu = 8
environment = 'phastcon'
readme_comment = f"Turning .wig files to pkl files"
""" Parameters: **CHANGE (AT)** """



def create_readme():
    name = os.path.join(data_dir, "readme")
    readme = open(name, "a")
    comment = readme_comment
    readme.write(comment)
    readme.close()


def gen_combination():
    create_readme()
    
    kind = name
    hash_obj = random.getrandbits(25)
    
    prg_file_path = os.path.join(job_path, get_file_name(kind= f"prg_{kind}_{hash_obj}"))
    slurm_file_path = os.path.join(job_path, get_file_name(kind= f"slurm_{kind}_{hash_obj}"))
    
    
    original_path = os.path.join(code_dir, main_script)
    os.system(f"cp {original_path} {data_dir}")

    python_script = os.path.join(data_dir, main_script)

    create_prg_file(prg_file_path, python_script)
    
    
    create_slurm_file(prg_file_path=prg_file_path, 
                    job_name=get_file_name(kind=f"{kind}_{hash_obj}", ext=False), 
                    slurm_file_path=slurm_file_path)


    ## (AT)
    os.system(f"chmod u+x {prg_file_path}")
    os.system(f"chmod u+x {slurm_file_path}")
    os.system(f"sbatch {slurm_file_path}")
for script in scripts_tobe_saved:
    script_dir = os.path.join(code_dir, script)
    os.system(f"cp {script_dir} {data_dir}")
                    

def main():
    gen_combination()

if __name__ == "__main__":
    main()


