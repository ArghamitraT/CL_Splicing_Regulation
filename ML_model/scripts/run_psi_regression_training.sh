# NOTES="No Contrastive Learning\nLinear Probing\nSGD optimizer\nFrozen encoder\nMean pooling\nSmall ResNet"
NOTES="Anshul data tisfm"

python -m scripts.psi_regression_training \
        task.global_batch_size=2048\
        task.pretraining_weights="exprmnt_2025_05_19__21_23_02"\
        trainer.max_epochs=30 \
        trainer.val_check_interval=0.5\
        embedder="tisfm"\
        tokenizer="onehot_tokenizer"\
        embedder.maxpooling=true\
        optimizer="sgd" \
        optimizer.lr=1e-3 \
        aux_models.freeze_encoder=false\
        aux_models.warm_start=true\
        logger.name="Psi__$(date +%Y%m%d_%H%M%S)" \
        logger.notes="$NOTES"
       
         
        