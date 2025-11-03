# "intronexon"  # or "3p", "5p", "intronOnly"
# aux_models.5p_weights: "exprmnt_2025_06_08__20_39_37"
# aux_models.3p_weights: "exprmnt_2025_06_08__20_38_28"
NOTES="trial"

# python -m scripts.psi_regression_training \
#         task.global_batch_size=2048\
#         trainer.max_epochs=10\
#         trainer.val_check_interval=0.5\
#         embedder="resnet"\
#         tokenizer="custom_tokenizer"\
#         embedder.maxpooling=true\
#         optimizer="sgd" \
#         optimizer.lr=1e-3 \
#         aux_models.freeze_encoder=false\
#         aux_models.warm_start=true\
#         aux_models.mtsplice_weights="exprmnt_2025_07_30__13_10_26"\
#         aux_models.mode="intronexon"\
#         aux_models.mtsplice_BCE=1\
#         logger.name="Psi_VE_mtspliceBCEResnetFrz0Wrm0__$(date +%Y%m%d_%H%M%S)" \
#         logger.notes="$NOTES"
       
         
# python -m scripts.psi_regression_training \
#         task.global_batch_size=2048\
#         trainer.max_epochs=2\
#         trainer.val_check_interval=0.5\
#         embedder="mtsplice"\
#         tokenizer="onehot_tokenizer"\
#         loss="MTSpliceBCELoss"\
#         embedder.maxpooling=true\
#         optimizer="adam" \
#         optimizer.lr=1e-3 \
#         aux_models.freeze_encoder=false\
#         aux_models.warm_start=true\
#         aux_models.dropout=0.8\
#         aux_models.mtsplice_weights="exprmnt_2025_10_26__14_30_11"\
#         aux_models.mode="mtsplice"\
#         aux_models.mtsplice_BCE=1\
#         dataset.fivep_ovrhang=300 \
#         dataset.threep_ovrhang=300 \
#         logger.name="Psi__trial__$(date +%Y%m%d_%H%M%S)" \
#         logger.notes="$NOTES"
       
# mode: or "3p", "5p", "intronOnly", "intronexon", "mtsplice"


# MAIN_DIR="/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data"
MAIN_DIR="/mnt/home/at3836/Contrastive_Learning/data/final_data/TSCelltype_finetuning"


# === Just specify file names ===
# TRAIN_FILE="train_3primeIntron_filtered.pkl"

TRAIN_FILE="psi_train_pericyte_psi_MERGED.pkl"
VAL_FILE="psi_val_pericyte_psi_MERGED.pkl"
TEST_FILE="psi_test_pericyte_psi_MERGED.pkl"


# # === Full paths constructed here ===
export TRAIN_DATA_FILE="${MAIN_DIR}/${TRAIN_FILE}"
export VAL_DATA_FILE="${MAIN_DIR}/${VAL_FILE}"
export TEST_DATA_FILE="${MAIN_DIR}/${TEST_FILE}"

python -m scripts.psi_regression_training \
        task.global_batch_size=2048\
        trainer.max_epochs=10\
        trainer.val_check_interval=0.5\
        embedder="mtsplice"\
        tokenizer="onehot_tokenizer"\
        loss="MTSpliceBCELoss"\
        loss.csv_dir=$MAIN_DIR\
        embedder.maxpooling=true\
        optimizer="adam" \
        optimizer.lr=1e-3 \
        aux_models.freeze_encoder=false\
        aux_models.warm_start=false\
        aux_models.dropout=0.8\
        aux_models.mtsplice_weights="exprmnt_2025_10_26__14_30_11"\
        aux_models.mode="mtsplice"\
        aux_models.mtsplice_BCE=1\
        dataset.fivep_ovrhang=300 \
        dataset.threep_ovrhang=300 \
        dataset.train_files.intronexon=$TRAIN_DATA_FILE\
        dataset.val_files.intronexon=$VAL_DATA_FILE\
        dataset.test_files.intronexon=$TEST_DATA_FILE\
        dataset.ascot=false \
        logger.name="Psi__trial__$(date +%Y%m%d_%H%M%S)" \
        logger.notes="$NOTES"

            