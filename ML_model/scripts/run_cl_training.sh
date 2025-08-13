


# === Set main data directory once ===

# MAIN_DIR="/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data"
MAIN_DIR="/mnt/home/nlk2136/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data"


# === Just specify file names ===
TRAIN_FILE="train_3primeIntron_filtered.pkl"
VAL_FILE="val_3primeIntron_filtered.pkl"
TEST_FILE="test_3primeIntron_filtered.pkl"

# TRAIN_FILE="train_ExonSeq_filtered.pkl"
# VAL_FILE="val_ExonSeq_filtered.pkl"
# TEST_FILE="test_ExonSeq_filtered.pkl"

# === Full paths constructed here ===
export TRAIN_DATA_FILE="${MAIN_DIR}/${TRAIN_FILE}"
export VAL_DATA_FILE="${MAIN_DIR}/${VAL_FILE}"
export TEST_DATA_FILE="${MAIN_DIR}/${TEST_FILE}"

# export CUDA_VISIBLE_DEVICES=1

<<<<<<< HEAD
NOTES="empire ai terminal test run"
=======
NOTES="termianl test run"
>>>>>>> f079f4b (Modify script for run on NYGC)

python -m scripts.cl_training \
        task=introns_cl \
        embedder="resnet101"\
        loss="supcon"\
        tokenizer="custom_tokenizer"\
        task.global_batch_size=2048\
        trainer.max_epochs=2 \
        trainer.val_check_interval=1.0\
        optimizer="sgd" \
        trainer.devices=1\
        logger.name="test_run$(date +%Y%m%d_%H%M%S)"\
        embedder.maxpooling=True\
        logger.notes="$NOTES"\
        dataset.n_augmentations=10 \
        dataset.train_data_file=$TRAIN_DATA_FILE \
        dataset.val_data_file=$VAL_DATA_FILE \
        dataset.test_data_file=$TEST_DATA_FILE


       
