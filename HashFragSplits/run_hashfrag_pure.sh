#!/bin/bash
#SBATCH --account=dklab
#SBATCH --job-name=hashFrag
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nkeung@nygenome.org
#SBATCH --output=/gpfs/commons/home/nkeung/hashFrag_CL/FULL_blastn_array_job.work/job_stdout_files/pure_hashFrag.jobid_%A_%a.out
#SBATCH --error=/gpfs/commons/home/nkeung/hashFrag_CL/FULL_blastn_array_job.work/job_stderr_files/pure_hashFrag.jobid_%A_%a.err

set -euo pipefail

set -e
cd $HOME
source ~/.bashrc
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate hash_frag_env

cd /gpfs/commons/home/nkeung/hashFrag/src/hashFrag/external/

# Directories and files
WORK_DIR="/gpfs/commons/home/nkeung/hashFrag_CL/FULL_blastn_array_job.work"
BLAST_DIR="$WORK_DIR/blast_results_single"

SW_SCRIPT="/gpfs/commons/home/nkeung/hashFrag/src/hashFrag/external/compute_blast_candidate_SW_scores.sh"

# Path to COMBINED train and test
FASTA_PATH="/gpfs/commons/home/nkeung/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/chromSplit/hashFragData/train_test_hg38_sequences_reverse.fa"

OUTPUT_PATH="$WORK_DIR/hashFrag_pure.blastn_candidates.sw_scores.tsv"

# Validate inputs
if [[ ! -f "$FASTA_PATH" ]]; then
    echo "ERROR: FASTA file not found: $FASTA_PATH"
    exit 1
fi

if [[ ! -f "$SW_SCRIPT" ]]; then
    echo "ERROR: SW script not found: $SW_SCRIPT"
    exit 1
fi

# Find partitioned BLAST files
shopt -s nullglob
PARTITIONS=("$BLAST_DIR"/*.tsv)

# Exclude previously generated SW output files
INPUT_PARTITIONS=()
for FILE in "${PARTITIONS[@]}"; do
    [[ "$FILE" == *.pairwise_scores.tsv ]] && continue
    INPUT_PARTITIONS+=("$FILE")
done

if [[ ${#INPUT_PARTITIONS[@]} -eq 0 ]]; then
    echo "ERROR: No input partitions found."
    exit 1
fi

echo "Found ${#INPUT_PARTITIONS[@]} BLAST partitions."

# Calculate exact SW scores
for PARTITION in "${INPUT_PARTITIONS[@]}"; do
    echo "Processing: $(basename "$PARTITION")"

    bash "$SW_SCRIPT" "$FASTA_PATH" "$PARTITION"
done

echo "SW calculations completed."

# Concatenate pairwise scores
echo "Concatenating SW score files..."

: > "$OUTPUT_PATH"

for PARTITION in "${INPUT_PARTITIONS[@]}"; do
    SCORE_FILE="${PARTITION%.tsv}.pairwise_scores.tsv"

    if [[ ! -f "$SCORE_FILE" ]]; then
        echo "ERROR: Missing expected output: $SCORE_FILE"
        exit 1
    fi

    cat "$SCORE_FILE" >> "$OUTPUT_PATH"
done

echo "Finished."
echo "Output: $OUTPUT_PATH"

echo "First 10 results:"
head -n 10 "$OUTPUT_PATH"

echo "Total SW score records:"
wc -l "$OUTPUT_PATH"
