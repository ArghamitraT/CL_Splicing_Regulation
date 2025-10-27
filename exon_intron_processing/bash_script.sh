#!/bin/bash
##ENVIRONMENT SETTINGS; REPLACE WITH CAUTION
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Simulation      #Set the job name too "JobExample1"
#SBATCH --time=4:00:00              #Set the wall clock limit to 1hr and 30min,takes 100min/EM iteration **CHANGE (AT)**
#SBATCH --mem=100G              
#SBATCH --cpus-per-task=2                   
#SBATCH --mail-type=END,FAIL    
#SBATCH --output=/gpfs/commons/home/atalukder/Contrastive_Learning/files/results/random_slurm_run/files/output_files/initital_data.%j      #Send stdout/err to
#SBATCH --mail-user=atalukder@nygenome.org 

set -e
cd $HOME
source ~/.bashrc
conda activate cl_splicing_regulation3
# conda activate phastcon
# gunzip -d /gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/refseq/*.fa.gz
# echo "unzipping completed!"



# --- Define Directory and Output Zip Name ---
# <<< IMPORTANT: REPLACE THESE PLACEHOLDERS >>>
DIRECTORY_TO_ZIP="/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash2"
OUTPUT_ZIP_FILE="/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash2.zip"
# <<< END OF PLACEHOLDERS >>>

# --- Change to the parent directory of the target directory (optional but often helpful) ---
PARENT_DIR=$(dirname "$DIRECTORY_TO_ZIP")
cd "$PARENT_DIR"
echo "Changed directory to: $(pwd)"

# --- Get just the directory name ---
DIR_NAME=$(basename "$DIRECTORY_TO_ZIP")

# --- Execute the zip command ---
echo "Starting zip operation..."
echo "Archiving directory: $DIR_NAME"
echo "Saving to: $OUTPUT_ZIP_FILE"

# Use -r for recursive, specify output file, then the directory name
zip -r "$OUTPUT_ZIP_FILE" "$DIR_NAME"

# Check the exit code of the zip command
ZIP_EXIT_CODE=$?
if [ $ZIP_EXIT_CODE -eq 0 ]; then
  echo "Zip operation completed successfully!"
else
  echo "Zip operation failed with exit code $ZIP_EXIT_CODE."
  exit $ZIP_EXIT_CODE # Exit script with the error code
fi

echo "Job finished."

# python /gpfs/commons/home/atalukder/Contrastive_Learning/code/exon_intron_processing/get_intronexon_seq3.py

# python /gpfs/commons/home/atalukder/Contrastive_Learning/code/exon_intron_processing/get_ExonIntron_position3_5and3prime2_RELIABLE.py

# python /gpfs/commons/home/atalukder/Contrastive_Learning/code/exon_intron_processing/get_ExonIntron_position3_5and3prime.py
# python /gpfs/commons/home/atalukder/Contrastive_Learning/code/exon_intron_processing/merge_split_TrainingValTest_Data.py
# python /gpfs/commons/home/atalukder/Contrastive_Learning/code/exon_intron_processing/get_intronexon_seq.py
# python /gpfs/commons/home/atalukder/Contrastive_Learning/code/get_geneAnnotFile.py
# python /gpfs/commons/home/atalukder/Contrastive_Learning/code/read_phastcon.py
# python /gpfs/commons/home/atalukder/Contrastive_Learning/code/get_intron_seq.py
# python /gpfs/commons/home/atalukder/Contrastive_Learning/code/get_ExonIntron_position2.py


# # URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz100way/alignments/knownGene.multiz100way.exonNuc.fa.gz"  # Replace with your actual URL
# URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/multiz100way/alignments/ncbiRefSeq.multiz100way.exonNuc.fa.gz"

# # The directory to save the downloaded file(s)
# OUTPUT_DIR="/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment"
# wget -P $OUTPUT_DIR $URL
# echo "Download completed!"

# 