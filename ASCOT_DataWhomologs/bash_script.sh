#!/bin/bash
##ENVIRONMENT SETTINGS; REPLACE WITH CAUTION
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Simulation      #Set the job name too "JobExample1"
#SBATCH --time=5:00:00              #Set the wall clock limit to 1hr and 30min,takes 100min/EM iteration **CHANGE (AT)**
#SBATCH --mem=80G              
#SBATCH --cpus-per-task=2                   
#SBATCH --mail-type=END,FAIL    
#SBATCH --output=/gpfs/commons/home/atalukder/Contrastive_Learning/files/results/random_slurm_run/files/output_files/initital_data.%j      #Send stdout/err to
#SBATCH --mail-user=atalukder@nygenome.org 

set -e
cd $HOME
source ~/.bashrc
conda activate cl_splicing_regulation3

python /gpfs/commons/home/atalukder/Contrastive_Learning/code/ASCOT_DataWhomologs/exon_exon_meanAbsDist.py

