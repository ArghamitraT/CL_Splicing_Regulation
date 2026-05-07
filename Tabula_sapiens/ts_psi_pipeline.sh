#!/bin/bash

set -e
cd $HOME
source ~/.bashrc
# Initialize conda for non-interactive shell
source $HOME/miniconda3/etc/profile.d/conda.sh

# Conda environment: Pandas, Numpy, PySam, 
conda activate tabula_sapiens
cd /gpfs/commons/home/nkeung/Contrastive_Learning/code/Tabula_sapiens

python verify_rmats.py                                                              # Ensure all cells successfully ran
python rmats_results.py --main_dir /gpfs/commons/home/nkeung/tabula_sapiens         # Calculate PSI from rMATS skipped junction events
python compile_final_data.py --main_dir /gpfs/commons/home/nkeung/tabula_sapiens

# Data Split: Exclude MULTIZ from training split. Randomly pick exons
python split_data/find_overlaps.py
python split_data/divide_train_val.py
# Final data saved in /gpfs/commons/home/nkeung/tabula_sapiens/psi_data/final_data

# Data Split: 
# Remove exons that appear in less than 10 cells. 
# Split by chromosome
python split_data/filtered_split.py
# Final data saved in /gpfs/commons/home/nkeung/tabula_sapiens/filtered_psi
