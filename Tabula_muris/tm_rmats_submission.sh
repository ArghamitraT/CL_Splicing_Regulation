#!/bin/bash

set -e
cd $HOME
source ~/.bashrc
# Initialize conda for non-interactive shell
source $HOME/miniconda3/etc/profile.d/conda.sh

# Conda environment: Pandas, Numpy, PySam, 
conda activate tabula_sapiens
cd /gpfs/commons/home/nkeung/Contrastive_Learning/code/Tabula_muris

python get_bam_paths.py                 # Basic metadata checks. Gets and saves paths to all BAM files
python check_cell_types.py              # Filters cell types with >= 30 observations

# Load samtools module (only for Tabula Sapiens)
# module load SAMtools/
python check_read_len.py                # Double check that all BAM paths have read lengths of 100
python submit_jobs.py                   # Submit all rMATS jobs. Note: Change script to update Slurm job submission and email

# All rMATS jobs submitted. Double check that these are completed. 
# Then run tm_psi_pipeline.sh
