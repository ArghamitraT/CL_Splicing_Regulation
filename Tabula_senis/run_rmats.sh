#!/bin/bash

##ENVIRONMENT SETTINGS; REPLACE WITH CAUTION
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=rMATS_pericyte_test            # MODIFY
#SBATCH --time=06:00:00                     #Set the wall clock limit -- MODIFY
#SBATCH --partition=cpu
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=output/out_rMATS_pericyte_%j            #Send stdout/err to -- MODIFY
#SBATCH --mail-user=nlk2136@columbia.edu

set -e
cd $HOME
source ~/.bashrc

# Initialize conda for non-interactive shell
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate rmats_testing

WORKDIR="/gpfs/commons/home/nkeung/tabula_sapiens/pericyte"
cd $WORKDIR
time rmats.py --gtf "/gpfs/commons/home/nkeung/gene_annotations/gencode.v45.primary_assembly.annotation.gtf" \
    --b1 b1.txt \
    --od output \
    --tmp data \
    -t single \
    --readLength 100 \
    --nthread 8 \
    --statoff