


# === Set main data directory once ===
MAIN_DIR="/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data"

# === Just specify file names ===
# TRAIN_FILE="train_3primeIntron_filtered.pkl"
# VAL_FILE="val_3primeIntron_filtered.pkl"
# TEST_FILE="test_3primeIntron_filtered.pkl"

TRAIN_FILE="train_ExonSeq_filtered.pkl"
VAL_FILE="val_ExonSeq_filtered.pkl"
TEST_FILE="test_ExonSeq_filtered.pkl"

# === Full paths constructed here ===
export TRAIN_DATA_FILE="${MAIN_DIR}/${TRAIN_FILE}"
export VAL_DATA_FILE="${MAIN_DIR}/${VAL_FILE}"
export TEST_DATA_FILE="${MAIN_DIR}/${TEST_FILE}"

export CUDA_VISIBLE_DEVICES=1

NOTES="try"

python -m scripts.cl_training \
        task=introns_cl \
        embedder="resnet"\
        tokenizer="custom_tokenizer"\
        task.global_batch_size=8192\
        trainer.max_epochs=5 \
        trainer.val_check_interval=1.0\
        optimizer="sgd" \
        trainer.devices=1\
        logger.name="cl_$(date +%Y%m%d_%H%M%S)"\
        embedder.maxpooling=True\
        logger.notes="$NOTES"\
        dataset.train_data_file=$TRAIN_DATA_FILE \
        dataset.val_data_file=$VAL_DATA_FILE \
        dataset.test_data_file=$TEST_DATA_FILE


       
         
# python -m scripts.cl_training \
#         task=introns_cl \
#         task.global_batch_size=8192\
#         trainer.max_epochs=5 \
#         trainer.val_check_interval=1.0\
#         tokenizer="custom_tokenizer" \
#         embedder="resnet" \
#         embedder.maxpooling=true\
#         optimizer="sgd" \
#         trainer.devices=1\
#         logger.name="cl_$(date +%Y%m%d_%H%M%S)"\
#         logger.notes="$NOTES"
