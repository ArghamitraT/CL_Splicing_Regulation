#!/bin/bash
##ENVIRONMENT SETTINGS; REPLACE WITH CAUTION
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Simulation      #Set the job name too "JobExample1"
#SBATCH --time=15:45:00              #Set the wall clock limit to 1hr and 30min,takes 100min/EM iteration **CHANGE (AT)**
#SBATCH --mem=50G              
#SBATCH --cpus-per-task=2                  
#SBATCH --mail-type=END,FAIL    
#SBATCH --output=/gpfs/commons/home/atalukder/Contrastive_Learning/files/results/random_slurm_rum/files/output_files/initital_data.%j      #Send stdout/err to
#SBATCH --mail-user=atalukder@nygenome.org 


find /gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/19_chr/wig_files -type f -exec sed -i 's/\(null\)/chr19/g' {} +
# set -e
# cd $HOME
# source ~/.bashrc
# # conda activate CntrstvLrn_1
# conda activate phastcon
# # gunzip -d /gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/refseq/*.fa.gz
# # echo "unzipping completed!"

# # python /gpfs/commons/home/atalukder/Contrastive_Learning/code/get_geneAnnotFile.py
# python /gpfs/commons/home/atalukder/Contrastive_Learning/code/run_phast/compare_phastconScore_WIGfiles.py




# python /gpfs/commons/home/atalukder/Contrastive_Learning/code/get_intron_seq.py
# python /gpfs/commons/home/atalukder/Contrastive_Learning/code/get_ExonIntron_position2.py


# # URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz100way/alignments/knownGene.multiz100way.exonNuc.fa.gz"  # Replace with your actual URL
# URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz100way/alignments/ncbiRefSeq.multiz100way.exonNuc.fa.gz"

# # The directory to save the downloaded file(s)
# OUTPUT_DIR="/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment"
# wget -P $OUTPUT_DIR $URL
# echo "Download completed!"

# 