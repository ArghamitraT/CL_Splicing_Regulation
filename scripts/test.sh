python -m scripts.test_lit_module \
        task=introns_cl \
        task.val_check_interval=1 \
        task.global_batch_size=8\
        trainer.devices=1 \
        trainer.max_epochs=10 \
        optimizer="sgd" \
        wandb.api_key="YOUR_API_KEY" \