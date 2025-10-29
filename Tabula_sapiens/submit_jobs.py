"""
This files is used to submit files in the slurm
"""
import os
import time
import pandas as pd
import json

# -------------------- CONFIG --------------------

hour = 2        # Modified in main. See below
memory = 128
nthred = 8

data_dir = "/gpfs/commons/home/nkeung/tabula_sapiens"
metadata_file = os.path.join(data_dir, "bam_paths.tsv")
cell_counts_file = os.path.join(data_dir, "cell_counts.tsv")
slurm_dir = os.path.join(data_dir, "jobs")

code_dir = "/gpfs/commons/home/nkeung/Contrastive_Learning/code/Tabula_sapiens"

# -------------------- CHECK PROGRESS --------------------

# Track completed rMATS jobs in JSON
json_file = os.path.join(data_dir, "completed.json")
if os.path.exists(json_file):
    with open(json_file, "r") as f:
        completed_cells = set(json.load(f))
else:
    completed_cells = set()

# -------------------- UTILITIES --------------------

def create_prg(cell_type):
   
    header = f"""#!/bin/bash
    set -e
    cd $HOME
    source ~/.bashrc

    # Generate b1.txt file for given cell_type
    conda activate tabula_sapiens
    WORKDIR={code_dir}
    cd $WORKDIR
    python compile_txt.py --cell_type "{cell_type}" --metadata {metadata_file} --main_dir {data_dir}
    conda deactivate

    # Run rMATS
    source $HOME/miniconda3/etc/profile.d/conda.sh
    conda activate rmats_testing
    WORKDIR={os.path.join(data_dir, cell_type.replace(" ", "_"))}
    cd $WORKDIR
    rmats.py --gtf "/gpfs/commons/home/nkeung/gene_annotations/gencode.v45.primary_assembly.annotation.gtf" \\
        --b1 b1.txt \\
        --od output \\
        --tmp data \\
        -t single \\
        --readLength 100 \\
        --nthread 8 \\
        --statoff"
    
    # Delete Temp and Zip Output
    WORKDIR={code_dir}
    cd $WORKDIR
    python post_process.py --cell_type "{cell_type}" --main_dir {data_dir}
    """

    return header
    


def create_slurm_file(prg_script, slurm_file_path, cell_type):
    safe_cell = cell_type.replace(" ", "_")
    timestamp = time.strftime("_%Y%m%d_%H%M%S")
    job_name = f"ts_{safe_cell}{timestamp}"
    header = f"#!/bin/bash\n" + \
    "##ENVIRONMENT SETTINGS; REPLACE WITH CAUTION\n" + \
    "##NECESSARY JOB SPECIFICATIONS\n" + \
    f"#SBATCH --job-name=ts_{job_name}      #Set the job name to \"JobExample1\"\n" + \
    "#SBATCH --partition=cpu     \n" + \
    f"#SBATCH --time={hour}:00:00              #Set the wall clock limit \n" + \
    f"#SBATCH --mem={memory}G              \n" + \
    f"#SBATCH --cpus-per-task={nthred}                   \n" + \
    "#SBATCH --mail-type=END,FAIL    \n" + \
    f"#SBATCH --output={slurm_dir}/output/out_{job_name}.%j      #Send stdout/err to\n" + \
    "#SBATCH --mail-user=nkeung@nygenome.org                    \n\n" + \
    f"{prg_script}"

    with open(slurm_file_path, "w") as f:
        f.write(header)
    return slurm_file_path


def submit_job(cell):
    
    prg_script = create_prg(cell) 
    
    safe_cell = cell.replace(" ", "_")
    slurm_file_path = os.path.join(slurm_dir, "scripts", f"ts_{safe_cell}.sh")

    create_slurm_file(prg_script=prg_script, 
                    slurm_file_path=slurm_file_path, 
                    cell_type = cell)

    os.system(f"chmod u+x {slurm_file_path}")
    os.system(f"sbatch {slurm_file_path}")
                    

def main():
    ts = pd.read_csv(metadata_file, sep="\t")
    cells = ts["cell_type"].unique().tolist()

    counts = pd.read_csv(cell_counts_file, sep="\t")
    for cell in cells:
        minutes_per_cell = 54 / 569  # runtime per cell in minutes
        num_cells = counts.loc[counts['cell_type'] == cell, 'count'].iloc[0]
        estimated_minutes = int(num_cells * minutes_per_cell * 1.2)  # add 20% buffer
        global hour
        hour = max(1, estimated_minutes // 60 + 1)  # round up to hours, min 1h

        safe_cell = cell.replace(" ", "_")
        if safe_cell in completed_cells:
            continue

        submit_job(cell)

if __name__ == "__main__":
    main()
