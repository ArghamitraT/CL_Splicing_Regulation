import os
import random
import time

# Generate a timestamp for unique folder and file names
timestamp = time.strftime("_%Y_%m_%d__%H_%M_%S")

# Create directories if they don't exist
def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

# Create the program file for phastCons
def create_phastcons_script(output_dir, target_coverage, expected_length, output_tree, maf_file, init_model):
    script_content = f"""#!/bin/bash
set -e
cd $HOME
source ~/.bashrc
conda activate CntrstvLrn_1
module load phast/1.5

phastCons --target-coverage {target_coverage} \\
    --expected-length {expected_length} \\
    --estimate-trees {output_tree} \\
    {maf_file} {init_model} --no-post-probs
"""
    script_path = os.path.join(output_dir, f"phastcons_job_{os.path.basename(maf_file)}.sh")
    with open(script_path, "w") as script_file:
        script_file.write(script_content)
    return script_path

# Create the SLURM file for submission
def create_slurm_file(script_path, job_name, slurm_output_dir):
    slurm_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time=5:45:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=8
#SBATCH --output={slurm_output_dir}/out_{job_name}.%j
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=atalukder@nygenome.org

bash {script_path}
"""
    slurm_path = os.path.join(slurm_output_dir, f"slurm_{job_name}.sh")
    with open(slurm_path, "w") as slurm_file:
        slurm_file.write(slurm_content)
    return slurm_path

# Submit jobs to SLURM
def submit_jobs(maf_files, output_dir, target_coverage, expected_length, init_model):
    # Create required directories
    job_output_dir = create_dir(os.path.join(output_dir, "job_outputs"))
    trees_output_dir = create_dir(os.path.join(output_dir, "trees"))
    
    for maf_file in maf_files:
        # Generate unique job name
        job_name = f"phastcons_{os.path.basename(maf_file).split('.')[0]}_{random.getrandbits(16)}"
        
        # Paths for script and SLURM files
        script_path = create_phastcons_script(
            output_dir=job_output_dir,
            target_coverage=target_coverage,
            expected_length=expected_length,
            output_tree=os.path.join(trees_output_dir, f"tree_{os.path.basename(maf_file)}"),
            maf_file=maf_file,
            init_model=init_model
        )
        
        slurm_path = create_slurm_file(script_path, job_name, job_output_dir)
        
        # Submit the job
        os.system(f"sbatch {slurm_path}")
        print(f"Submitted job: {job_name}")

# Main function
def main():
    # Parameters
    maf_dir = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/chr21"
    init_model = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/trees/hg19.100way.phastCons.mod"
    output_dir = f"/gpfs/commons/home/atalukder/Contrastive_Learning/files/results/phastcons_jobs{timestamp}"
    target_coverage = 0.3
    expected_length = 45
    
    # Gather all MAF files
    maf_files = [os.path.join(maf_dir, f) for f in os.listdir(maf_dir) if f.endswith(".maf")]
    
    # Submit jobs
    submit_jobs(maf_files, output_dir, target_coverage, expected_length, init_model)

if __name__ == "__main__":
    main()
