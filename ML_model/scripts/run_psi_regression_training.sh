# NOTES="No Contrastive Learning\nLinear Probing\nSGD optimizer\nFrozen encoder\nMean pooling\nSmall ResNet"
NOTES="Psi\nRanom initialization baseline"

python -m scripts.psi_regression_training \
        task.global_batch_size=4096\
        trainer.max_epochs=5 \
        trainer.val_check_interval=0.5\
        tokenizer="custom_tokenizer" \
        embedder="resnet" \
        optimizer="sgd" \
        optimizer.lr=1e-3 \
        logger.name="Psi_$(date +%Y%m%d_%H%M%S)" \
        aux_models.freeze_encoder=false\
        aux_models.warm_start=false\
        trainer.devices=1\
        logger.notes="$NOTES"
       
         
        