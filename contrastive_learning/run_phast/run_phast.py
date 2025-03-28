# phastCons --target-coverage 0.3 --expected-length 45 --estimate-trees trees/newtrees_chr1 maf_files/chr1.maf trees/hg19.100way.phastCons.mod --no-post-probs


"""
This files is used to submit files in the slurm
"""
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

def create_prg_file(prg_file_path, target_coverage, exp_len, ouput_tree_file, msa_file, input_tre_file):
   
    # header = f"#!/bin/bash\n" + \
    # "set -e\n" + \
    # "cd $HOME\n" + \
    # "source ~/.bashrc\n" + \
    # "conda activate CntrstvLrn_1\n" + \
    # "module load phast/1.5\n" + \
    # f"phastCons --target-coverage {target_coverage} --expected-length {exp_len} --estimate-trees {ouput_tree_file} {msa_file} {input_tre_file} --no-post-probs"
    
    header = f"#!/bin/bash\n" + \
    "set -e\n" + \
    "cd $HOME\n" + \
    "source ~/.bashrc\n" + \
    "conda activate CntrstvLrn_1\n" + \
    "module load phast/1.5\n" + \
    f"phastCons --target-coverage {target_coverage} --expected-length {exp_len}  --estimate-rho {ouput_tree_file} {msa_file} {input_tre_file} --no-post-probs"
 

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
    "#SBATCH --time=50:45:00              #Set the wall clock limit to 1hr and 30min, # takes 100min/EM iteration **CHANGE (AT)**\n" + \
    "#SBATCH --mem=100G              \n" + \
    "#SBATCH --cpus-per-task=20                   \n" + \
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


samples_file_names = [['ds_10_num1_aln_11_long', 'ds_100_num1_aln_01_short']]



""" **CHANGE (AT)** THE DIRECTORY NAME FROM WHERE WEIGHTS NEEDS TO BE COPIED INTO ./WEIGHTS FOLDER(THE UNIVERSAL WEIGHT FOLDER)"""
input_data_dir = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/"
target_coverage = 0.3
exp_len = 45
ouput_tree_file = output_dir +'/newtrees_chr19'
msa_file = input_data_dir+'19_chr/chr19.maf'
input_tre_file = input_data_dir+'trees/hg19.100way.phastCons.mod'
name = "run_phast"


def create_readme():
    name = os.path.join(data_dir, "readme")
    readme = open(name, "a")

    """ **CHANGE (AT)** WRITE THE COMMENT"""
    comment = f"Running phastcon, chrm 19, estimate rho instead of estimate tree, seeing if --cpus-per-task increase (10 to 20) reduces time."
    readme.write(comment)
    readme.close()


def gen_combination():
    create_readme()

   
    kind = name
    hash_obj = random.getrandbits(25)
    
    prg_file_path = os.path.join(job_path, get_file_name(kind= f"prg_{kind}_{hash_obj}"))
    slurm_file_path = os.path.join(job_path, get_file_name(kind= f"slurm_{kind}_{hash_obj}"))
    
    create_prg_file(prg_file_path, target_coverage, exp_len, ouput_tree_file, msa_file, input_tre_file)
    
    
    create_slurm_file(prg_file_path=prg_file_path, 
                    job_name=get_file_name(kind=f"{kind}_{hash_obj}", ext=False), 
                    slurm_file_path=slurm_file_path)


    os.system(f"cp {prg_file_path} {data_dir}")
    os.system(f"cp {slurm_file_path} {data_dir}")
    print()
    ## (AT)
    os.system(f"chmod u+x {prg_file_path}")
    os.system(f"chmod u+x {slurm_file_path}")
    os.system(f"sbatch {slurm_file_path}")
    
                    

def main():
    gen_combination()

if __name__ == "__main__":
    main()


