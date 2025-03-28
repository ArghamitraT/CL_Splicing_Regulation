# """
# script: convet_wigtoBw.sh

# Description:
# ------------
# This script converts `.wig` files to `.bw` (BigWig) files using Wiggletools. It takes all `.wig` files from the input directory, processes each file, and stores the resulting `.bw` files in the output directory.

# Inputs:
# -------
# - `INPUT_DIR`: Directory containing `.wig` files to process.
#   Example: `/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/19_chr/wig_files/`

# Outputs:
# --------
# - `.bw` files corresponding to each `.wig` file, stored in the `OUTPUT_DIR`.
#   Example: `/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/19_chr/bw_files/`

# Workflow:
# ---------
# 1. Loops through all `.wig` files in the input directory.
# 2. Converts each `.wig` file to `.bw` using Wiggletools.
# 3. Stores the resulting `.bw` file in the output directory.

# Usage Example:
# --------------
# 1. Set the paths for `INPUT_DIR` and `OUTPUT_DIR` appropriately.
# 2. Submit the job script using `sbatch`:
#    sbatch bwtowig_conversion.sh
# """


set -e
cd $HOME
source ~/.bashrc
# conda activate CntrstvLrn_1
conda activate phastcon

# Define input and output directories
INPUT_DIR="/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/19_chr/wig_files"
OUTPUT_DIR="/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/19_chr/bw_files"
CHROM_SIZES="/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/hg19.chrom.sizes"  # Replace with actual path to chrom.sizes

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Loop through all .wig files in the input directory
for WIG_FILE in "$INPUT_DIR"/*.wig; do
  # Extract the base name of the file (without extension)
  BASE_NAME=$(basename "$WIG_FILE" .wig)

  # Construct the output BigWig file path
  BW_FILE="$OUTPUT_DIR/${BASE_NAME}.bw"

  # Convert WIG to BigWig
  wigToBigWig "$WIG_FILE" "$CHROM_SIZES" "$BW_FILE"

  # wiggletools write "$BW_FILE" "$WIG_FILE"
#   wigToBigWig "$WIG_FILE" "$CHROM_SIZES" "$BW_FILE"

  echo "Converted $WIG_FILE to $BW_FILE"
done


#  wigToBigWig /gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/19_chr/wig_files/score_chr19_hg19_195_doubled.wig /gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/hg19.chrom.sizes /gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/19_chr/bw_files_wigToBigwick/score_chr19_hg19_195_doubled.bw