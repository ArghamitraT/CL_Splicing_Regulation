#!/bin/bash
#SBATCH --account=dklab
#SBATCH --job-name=hashFrag
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --array=0-51%10
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nkeung@nygenome.org
#SBATCH --output=/gpfs/commons/home/nkeung/hashFrag_CL/FULL_blastn_array_job.work/job_stdout_files/pure_hashFrag.jobid_%A_%a.out
#SBATCH --error=/gpfs/commons/home/nkeung/hashFrag_CL/FULL_blastn_array_job.work/job_stderr_files/pure_hashFrag.jobid_%A_%a.err

set -euo pipefail

cd $HOME
source ~/.bashrc
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate hash_frag_env

cd /gpfs/commons/home/nkeung/hashFrag/src/hashFrag/external/

# Directories and files
WORK_DIR="/gpfs/commons/home/nkeung/hashFrag_CL/FULL_blastn_array_job.work"
BLAST_DIR="$WORK_DIR/blast_results"

SW_SCRIPT="/gpfs/commons/home/nkeung/hashFrag/src/hashFrag/external/compute_blast_candidate_SW_scores.sh"

# Path to COMBINED train and test
FASTA_PATH="/gpfs/commons/home/nkeung/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/chromSplit/hashFragData/train_test_hg38_sequences_reverse.fa"

# Validate inputs
if [[ ! -f "$FASTA_PATH" ]]; then
    echo "ERROR: FASTA file not found: $FASTA_PATH"
    exit 1
fi

if [[ ! -f "$SW_SCRIPT" ]]; then
    echo "ERROR: SW script not found: $SW_SCRIPT"
    exit 1
fi


# Select this array task's partition (IDs 1–52)
TASK_ID=$((SLURM_ARRAY_TASK_ID + 1))

PARTITION="$BLAST_DIR/test_hg38_sequences.part_$(printf '%08d' "$TASK_ID").blastn.processed.tsv"

if [[ ! -f "$PARTITION" ]]; then
    echo "ERROR: Partition not found: $PARTITION"
    exit 1
fi

# Avoid overwriting existing result
OUTPUT_FILE="${PARTITION%.tsv}.pairwise_scores.tsv"
if [[ -f "$OUTPUT_FILE" ]]; then
    echo "ERROR: Output already exists: $OUTPUT_FILE"
    echo "Skipping to avoid overwriting existing results."
    exit 1
fi

echo "Array task: $TASK_ID"
echo "Processing: $(basename "$PARTITION")"
date

# Calculate exact SW scores for this partition only
bash "$SW_SCRIPT" "$FASTA_PATH" "$PARTITION"

echo "SW calculation completed."
echo "Output: ${PARTITION%.tsv}.pairwise_scores.tsv"
date
