python -m scripts.cl_training \
        task=introns_cl \
        task.val_check_interval=64 \
        task.global_batch_size=64\
        dataset.num_workers=4 \
        trainer.max_epochs=10 \
        tokenizer="hf_tokenizer" \
        embedder="ntv2" \
        optimizer="sgd" \