python -m scripts.cl_training \
        task=introns_cl \
        task.val_check_interval=512\
        task.global_batch_size=512\
        dataset.num_workers=4 \
        trainer.max_epochs=2 \
        tokenizer="hf_tokenizer" \
        embedder="ntv2" \
        optimizer="sgd" \
        # logger.name="Argha"
         # trainer.devices=1\
        