# NOTES="No Contrastive Learning\nLinear Probing\nSGD optimizer\nFrozen encoder\nMean pooling\nSmall ResNet"
NOTES="Anshul data"

python -m scripts.psi_regression_training \
        task.global_batch_size=2048\
        trainer.max_epochs=5 \
        trainer.val_check_interval=0.5\
        tokenizer="custom_tokenizer" \
        embedder="resnet" \
        embedder.maxpooling=true\
        optimizer="sgd" \
        optimizer.lr=1e-3 \
        aux_models.freeze_encoder=false\
        aux_models.warm_start=false\
        logger.name="Psi__$(date +%Y%m%d_%H%M%S)"\
        logger.notes="$NOTES"
       
         
        