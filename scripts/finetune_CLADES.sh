################################################################################
# README — Fine-Tuning Script Specifications
#
# This script launches the PSI regression / ELRC fine-tuning pipeline for CLADES,
# using a pretrained MTSplice-style encoder and species-aware intronic sequence
# representations.
#
# ============================
# MODEL COMPONENTS
# ============================
# • Pretrained CLADES encoder (MTSplice backbone)
# • Optionally frozen or trainable encoder
# • Auxiliary PSI regression head with configurable dropout
# • Warm-starting from a chosen checkpoint (pretrain → finetune)
# • Optional evaluation-only mode using a fixed checkpoint
#
# ============================
# INPUT DATA REQUIREMENTS
# ============================
# The dataset must contain:
#   • exon_id
#   • 5′ intron window (default: 300 bp)
#   • 3′ intron window (default: 300 bp)
#   • tissue- or cell-type-specific PSI targets
#   • (Optional) mask for missing PSI values
#
# The Hydra config expects:
#   dataset.fivep_ovrhang = 300    # 5′ intron window length
#   dataset.threep_ovrhang = 300   # 3′ intron window length
#
# Make sure your dataset directory is correctly set inside the Hydra config
# (or passed explicitly during script invocation).
#
# ============================
# PRETRAINED WEIGHT REQUIREMENTS
# ============================
# The following parameters must point to existing checkpoint directories:
#
#   aux_models.mtsplice_weights = "<pretrain_run_folder>"
#       → Folder containing the pretrained CLADES encoder weights.
#
#   aux_models.eval_weights = "<finetuned_run_folder>"
#       → Folder containing a previous finetuned checkpoint (if eval mode).
#
# Warm-start behavior:
#   • aux_models.warm_start=true: Load encoder + task head from pretraining.
#   • aux_models.freeze_encoder=false: Allow encoder weights to update.
#   • aux_models.train_mode="eval": Run evaluation only (no training).
#
# ============================
# HYDRA CONFIG OVERRIDES
# ============================
# Example overrides used in this script:
#
#   • task.global_batch_size=2048      → Global effective batch size
#   • trainer.max_epochs=2             → Number of fine-tuning epochs
#   • optimizer.lr=1e-3                → Learning rate for fine-tuning
#   • aux_models.freeze_encoder=false  → Unfreeze encoder for adaptation
#   • aux_models.warm_start=true       → Load pretrained encoder weights
#   • aux_models.dropout=0.8           → Dropout in PSI regression head
#   • aux_models.train_mode="eval"     → Evaluation mode (skip training)
#
# ============================
# LOGGING (W&B)
# ============================
# Set run name and notes:
#
#   logger.name="Psi__trial__<timestamp>"
#   logger.notes="$NOTES"
#
# NOTES can be used to annotate the run:
#   NOTES="abl_no_freeze"   # e.g., ablation: no frozen layers
#   NOTES="full_finetune"
#
# ============================
# RUNNING THE SCRIPT
# ============================
# You can run fine-tuning either through this bash launcher:
#
#   bash finetune_CLADES.sh
#
# Or directly:
#
#   python -m finetune_CLADES \
#       task.global_batch_size=2048 \
#       trainer.max_epochs=2 \
#       optimizer.lr=1e-3 \
#       aux_models.warm_start=true \
#       ...
#
################################################################################



NOTES="trial"

         
python -m finetune_CLADES \
        task.global_batch_size=2048\
        trainer.max_epochs=2\
        optimizer.lr=1e-3 \
        aux_models.freeze_encoder=false\
        aux_models.warm_start=true\
        aux_models.dropout=0.8\
        aux_models.mtsplice_weights="pretrain_2025_11_14_23_12_22"\
        aux_models.train_mode="train"\
        aux_models.eval_weights="finetune_2025_11_14_23_46_21"\
        dataset.fivep_ovrhang=300 \
        dataset.threep_ovrhang=300 \
        logger.name="Psi__trial__$(date +%Y%m%d_%H%M%S)" \
        logger.notes="$NOTES"
       