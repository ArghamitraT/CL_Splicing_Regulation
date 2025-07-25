# "intronexon"  # or "3p", "5p", "intronOnly"
# aux_models.5p_weights: "exprmnt_2025_06_08__20_39_37"
# aux_models.3p_weights: "exprmnt_2025_06_08__20_38_28"
NOTES="supcon loss 2 aug resnet 101"

python -m scripts.psi_regression_training \
        task.global_batch_size=2048\
        trainer.max_epochs=15\
        trainer.val_check_interval=0.5\
        embedder="resnet101"\
        tokenizer="custom_tokenizer"\
        embedder.maxpooling=true\
        optimizer="sgd" \
        optimizer.lr=1e-3 \
        aux_models.freeze_encoder=false\
        aux_models.warm_start=true\
        aux_models.mode="3p"\
        logger.name="Psi_supcon10augAllrestnet101__$(date +%Y%m%d_%H%M%S)" \
        logger.notes="$NOTES"
       
         
        