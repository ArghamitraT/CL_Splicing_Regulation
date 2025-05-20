NOTES="interpretable encoder trying in empireAI lastime did not work"

python -m scripts.cl_training \
        task=introns_cl \
        embedder="tisfm"\
        tokenizer="onehot_tokenizer"\
        task.global_batch_size=8192\
        trainer.max_epochs=2 \
        trainer.val_check_interval=1.0\
        optimizer="sgd" \
        trainer.devices=1\
        logger.name="cl_$(date +%Y%m%d_%H%M%S)"\
        embedder.maxpooling=True\
        logger.notes="$NOTES"

       
         
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
