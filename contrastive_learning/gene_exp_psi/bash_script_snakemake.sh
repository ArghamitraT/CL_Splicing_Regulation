#!/bin/bash
##ENVIRONMENT SETTINGS; REPLACE WITH CAUTION
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Gene_exp_psi      #Set the job name too "JobExample1"
#SBATCH --time=10:45:00              #Set the wall clock limit to 1hr and 30min,takes 100min/EM iteration **CHANGE (AT)**
#SBATCH --mem=128G              
#SBATCH --cpus-per-task=18                   
#SBATCH --mail-type=END,FAIL    
#SBATCH --output=/gpfs/commons/home/atalukder/Contrastive_Learning/files/results/random_slurm_rum/files/output_files/initital_data.%j      #Send stdout/err to
#SBATCH --mail-user=atalukder@nygenome.org 

set -e
cd $HOME
source ~/.bashrc
conda activate snakemake_env
# module load R/4.3.3
# module load snakemake/7.30.1
# snakemake -s /gpfs/commons/home/atalukder/Contrastive_Learning/models/gene_exp_psi/Snakefile2 --cores 16
# Rscript /gpfs/commons/home/atalukder/Contrastive_Learning/models/gene_exp_psi/create_signatures2.R
# gunzip -d /gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/refseq/*.fa.gz
# echo "unzipping completed!"

# python /gpfs/commons/home/atalukder/Contrastive_Learning/models/gene_exp_psi/create_signatures_optimized.py
# python /gpfs/commons/home/atalukder/Contrastive_Learning/code/match_exon_intron_boundary2.py
# python /gpfs/commons/home/atalukder/Contrastive_Learning/models/gene_exp_psi/try.py
python /gpfs/commons/home/atalukder/Contrastive_Learning/code/gene_exp_psi/get_psi_val.py

# # URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz100way/alignments/knownGene.multiz100way.exonNuc.fa.gz"  # Replace with your actual URL
# URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz100way/alignments/ncbiRefSeq.multiz100way.exonNuc.fa.gz"

# # The directory to save the downloaded file(s)
# OUTPUT_DIR="/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment"
# wget -P $OUTPUT_DIR $URL
# echo "Download completed!"

# 