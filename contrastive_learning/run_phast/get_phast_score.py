"""
Script: get_phast_score.py

Description:
This script automates the creation and submission of SLURM jobs for running `phastCons` to calculate conservation scores.
The script handles:
1. Generating necessary job and program files for each non-conserved tree in the input directory.
2. Submitting the generated jobs to a SLURM scheduler.
3. Organizing outputs into a structured directory.

Inputs:
- Input files:
  - `msa_file`: Multiple sequence alignment (MAF) file.
  - `conserved_tree_file`: File specifying the conserved phylogenetic tree.
  - `non_conserved_tree_folder`: Directory containing non-conserved tree files.
- Parameters:
  - `target_coverage`: Coverage parameter for `phastCons`.
  - `exp_len`: Expected length parameter for `phastCons`.
  - SLURM job specifications (e.g., memory, CPU, wall time).

Outputs:
1. Program files (`.sh`) and SLURM submission files for each non-conserved tree.
2. Output `.wig` files containing conservation scores for each modified tree.
3. Log files and saved scripts for reproducibility.

Workflow:
1. **Directory Setup**:
   - Creates directories to store outputs, weights, and job files.
2. **Job File Generation**:
   - For each non-conserved tree file:
     - Generates a `.sh` file with the `phastCons` command for the specific tree.
     - Generates a corresponding SLURM `.sh` file for job submission.
3. **Job Submission**:
   - Makes the files executable and submits the SLURM job.
4. **Logging**:
   - Copies relevant scripts to the output directory for reproducibility.
   - Creates a `readme` file to document job details.

Dependencies:
- Requires `phast/1.5` module to be loaded for running `phastCons`.
- Uses a Python environment with `os`, `time`, and `random` libraries.

Usage:
Run the script directly:
   python phastCons_job_submission.py

Ensure to update the `Parameters: **CHANGE (AT)**` section to match your experiment's settings before execution.
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

def create_prg_file(prg_file_path, target_coverage, exp_len, msa_file, non_conserved_tree_file, output_wig_file):
   
    header = f"#!/bin/bash\n" + \
    "set -e\n" + \
    "cd $HOME\n" + \
    "source ~/.bashrc\n" + \
    "conda activate CntrstvLrn_1\n" + \
    "module load phast/1.5\n" + \
    f"phastCons --target-coverage {target_coverage} --expected-length {exp_len} {msa_file} {conserved_tree_file},{non_conserved_tree_file} > {output_wig_file}"
 
    # f"phastCons --target-coverage {target_coverage} --expected-length {exp_len}  --estimate-rho {ouput_tree_file} {msa_file} {input_tre_file} --no-post-probs"
 

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


""" Parameters: **CHANGE (AT)** """
input_data_dir = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/"
code_dir = '/gpfs/commons/home/atalukder/Contrastive_Learning/code/run_phast'
target_coverage = 0.3
exp_len = 45
msa_file = input_data_dir+'19_chr/chr19.maf'
conserved_tree_file = input_data_dir+'19_chr/TREES/newtrees_chr19.cons.mod'
non_conserved_tree_folder = input_data_dir+'19_chr/TREES/modified_trees/'
scripts_tobe_saved= ['get_phast_score.py']
name = "run_phast"
hour = 10 # hour
memory = 50 #GB
readme_comment = f"Running phastcon, chrm 19, gettting the score manupulating each branch at a time."
""" Parameters: **CHANGE (AT)** """



def create_readme():
    name = os.path.join(data_dir, "readme")
    readme = open(name, "a")
    comment = readme_comment
    readme.write(comment)
    readme.close()


def gen_combination():
    create_readme()

    file_list = [f for f in os.listdir(non_conserved_tree_folder) if os.path.isfile(os.path.join(non_conserved_tree_folder, f))]

    for non_conserved_tree in file_list:
        kind = name
        hash_obj = random.getrandbits(25)
        
        prg_file_path = os.path.join(job_path, get_file_name(kind= f"prg_{kind}_{hash_obj}"))
        slurm_file_path = os.path.join(job_path, get_file_name(kind= f"slurm_{kind}_{hash_obj}"))
        
        output_choromSpecific_name = non_conserved_tree.rsplit('.', 1)[0]
        output_wig_file = output_dir + f'/score_chr19_{output_choromSpecific_name}.wig'
        non_conserved_tree_file = os.path.join(non_conserved_tree_folder, non_conserved_tree)
        create_prg_file(prg_file_path, target_coverage, exp_len, msa_file, non_conserved_tree_file, output_wig_file)
        
        
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


