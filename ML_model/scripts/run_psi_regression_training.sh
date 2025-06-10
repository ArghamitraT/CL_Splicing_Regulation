# NOTES="No Contrastive Learning\nLinear Probing\nSGD optimizer\nFrozen encoder\nMean pooling\nSmall ResNet"
NOTES="try"

python -m scripts.psi_regression_training \
        task.global_batch_size=2048\
        trainer.max_epochs=15\
        trainer.val_check_interval=0.5\
        embedder="resnet"\
        tokenizer="custom_tokenizer"\
        embedder.maxpooling=true\
        optimizer="sgd" \
        optimizer.lr=1e-3 \
        aux_models.freeze_encoder=false\
        aux_models.warm_start=true\
        aux_models.mode="intronexon"\
        logger.name="Psi_intronexonExnWrm__$(date +%Y%m%d_%H%M%S)" \
        logger.notes="$NOTES"
       
         
        