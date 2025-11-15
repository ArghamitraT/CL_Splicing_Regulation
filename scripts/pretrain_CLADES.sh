
################################################################################
# README — Training Script Specifications
#
# This script launches the contrastive pretraining pipeline for CLADES using:
#   • MTSplice encoder (dual-branch CNN for 5' and 3' boundaries)
#   • One-hot tokenizer
#   • Weighted supervised contrastive loss
#   • Multi-species intron/exon alignment dataset (Min ≥ 30 views)
#
# ============================
# INPUT DATA REQUIREMENTS
# ============================
# The script expects three pickled files, each containing:
#   • exon_id
#   • species views (≥ 30 aligned intronic views)
#   • 5′ and 3′ intron sequence windows (one-hot encoded or raw string)
#   • metadata (species labels, alignment mask, etc.)
#
# Files must exist under the directory structure:
#
#   MAIN_DIR/
#     ├── train_merged_filtered_min30Views.pkl
#     ├── val_merged_filtered_min30Views.pkl
#     └── test_merged_filtered_min30Views.pkl
#
# Set MAIN_DIR to point to your processed multiz-alignment dataset.
#
# ============================
# HYDRA CONFIG OVERRIDES
# ============================
# Key Hydra overrides used in this script:
#   • task=introns_cl                 => Enables CLADES contrastive pipeline
#   • embedder="mtsplice"             => Use MTSplice-style encoder
#   • tokenizer="onehot_tokenizer"    => One-hot 5'/3' intron boundaries
#   • loss="supcon"                   => Supervised contrastive objective
#   • dataset.n_augmentations=2       => Two positive views per anchor (species)
#   • trainer.max_epochs=25           => Training epochs
#   • task.global_batch_size=2048     => Global batch size across devices
#
# ============================
# LOGGING (W&B)
# ============================
# logger.name="cl_trial_<timestamp>"
# logger.notes="$NOTES"
#
# Notes allow tagging the run with a short comment for experiment tracking.
#
# ============================
# RUNNING THE SCRIPT
# ============================
#
#   bash pretrain_CLADES.sh
#
# or directly:
#
#   python -m pretrain_CLADES.py
#
################################################################################


#!/bin/bash

# ========= Locate CLADES root directory ========= #
find_contrastive_root() {
    local start_dir
    start_dir=$(dirname "$(readlink -f "$0")")

    local dir="$start_dir"
    while [[ "$dir" != "/" ]]; do
        if [[ "$(basename "$dir")" == "CLADES" ]]; then
            echo "$dir"
            return 0
        fi
        dir=$(dirname "$dir")
    done

    echo "ERROR: Could not find CLADES directory." >&2
    exit 1
}

export CONTRASTIVE_ROOT="$(find_contrastive_root)"
echo "CONTRASTIVE_ROOT = $CONTRASTIVE_ROOT"

# ========= DATA DIRECTORY ========= #
MAIN_DIR="${CONTRASTIVE_ROOT}/data/pretrain_sample_data"
echo "MAIN_DIR = $MAIN_DIR"


# ========= File names ========= #
TRAIN_FILE="train_merged_filtered_min30Views.pkl"
VAL_FILE="val_merged_filtered_min30Views.pkl"
TEST_FILE="test_merged_filtered_min30Views.pkl"

# ========= Construct full paths ========= #
export TRAIN_DATA_FILE="${MAIN_DIR}/${TRAIN_FILE}"
export VAL_DATA_FILE="${MAIN_DIR}/${VAL_FILE}"
export TEST_DATA_FILE="${MAIN_DIR}/${TEST_FILE}"

echo "TRAIN_DATA_FILE = $TRAIN_DATA_FILE"
echo "VAL_DATA_FILE   = $VAL_DATA_FILE"
echo "TEST_DATA_FILE  = $TEST_DATA_FILE"


NOTES="try"

# === Your python launch ===
python -m pretrain_CLADES \
        task=introns_cl \
        embedder="mtsplice"\
        loss="supcon"\
        tokenizer="onehot_tokenizer"\
        task.global_batch_size=2048\
        trainer.max_epochs=1 \
        optimizer="adam" \
        logger.name="cl_trial_$(date +%Y%m%d_%H%M%S)"\
        logger.notes="$NOTES"\
        dataset.n_augmentations=2 \
        dataset.train_data_file=$TRAIN_DATA_FILE \
        dataset.val_data_file=$VAL_DATA_FILE \
        dataset.test_data_file=$TEST_DATA_FILE \
       