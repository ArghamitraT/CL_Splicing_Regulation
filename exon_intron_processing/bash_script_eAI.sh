#!/bin/bash
##ENVIRONMENT SETTINGS; REPLACE WITH CAUTION
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Unzip_IntronExon   # Set a specific job name
#SBATCH --partition=columbia          # eai specific partition
#SBATCH --account=columbia            # eai specific account
#SBATCH --time=5:30:00                # Set wall clock limit (30 min should be enough)
#SBATCH --mem=100G                      # Memory request (5G is likely sufficient)
#SBATCH --cpus-per-task=2             # Number of CPUs (unzip is usually single-threaded)
#SBATCH --mail-type=END,FAIL          # Email notifications
#SBATCH --output=/mnt/home/at3836/Contrastive_Learning/files/results/random_slurm_run/files/output_files/unzip_intronexon.%j # Log file path
#SBATCH --mail-user=at3836@columbia.edu # Your Columbia email

# --- Load necessary modules and activate environment ---
echo "Loading environment..."
set -e            # Exit immediately if a command exits with a non-zero status.
# cd $HOME        # Go to home directory (optional, often handled by Slurm/bashrc)
source ~/.bashrc  # Source your bash profile to potentially load conda
conda activate cl_splicing_regulation3 # Activate your conda environment
echo "Conda environment activated: cl_splicing_regulation3"

# --- Define the target directory and file ---
TARGET_DIR="/mnt/home/at3836/Contrastive_Learning/data/final_data_new/intronExonSeq_multizAlignment_noDash"
ZIP_FILE="intronExonSeq_multizAlignment_noDash2.zip"
FULL_ZIP_PATH="${TARGET_DIR}/${ZIP_FILE}"

echo "Target directory: $TARGET_DIR"
echo "Zip file: $FULL_ZIP_PATH"

# --- Change to the target directory ---
cd "$TARGET_DIR"
echo "Changed directory to: $(pwd)"

# --- Unzip the file ---
echo "Starting unzip operation for $ZIP_FILE..."

# Check if the zip file exists before attempting to unzip
if [ -f "$ZIP_FILE" ]; then
    unzip "$ZIP_FILE"
    # Check the exit code of unzip
    UNZIP_EXIT_CODE=$?
    if [ $UNZIP_EXIT_CODE -eq 0 ]; then
        echo "Unzip operation completed successfully!"
        # Optional: Remove the zip file after successful extraction
        # echo "Removing zip file..."
        # rm "$ZIP_FILE"
    else
        echo "Unzip operation failed with exit code $UNZIP_EXIT_CODE."
        exit $UNZIP_EXIT_CODE # Exit script with the error code
    fi
else
    echo "Error: Zip file not found at $FULL_ZIP_PATH"
    exit 1 # Exit with an error code
fi

echo "Job finished."
