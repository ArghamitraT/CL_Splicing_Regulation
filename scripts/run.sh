#python -m scripts.cl_training \
#        task=introns_cl \
#        task.val_check_interval=64 \
#        task.global_batch_size=512\
#        dataset.num_workers=8 \
#        dataset.batch_size_per_device=64 \
#       trainer.max_epochs=10 \
#        trainer.devices=2 \
#        tokenizer="hf_tokenizer" \
#        embedder="ntv2" \
#        optimizer="adamw" \
#        optimizer.fn.lr=0.001 \
#        logger.project="INTRONS_CL" \
#        wandb.api_key="765bad652bcb6ce569641fc334bcf0f0eea5b1fb" \

#python -m scripts.cl_training \
#        task=introns_cl \
#        task.val_check_interval=64 \
#        task.global_batch_size=512\
#        dataset.num_workers=8 \
#        dataset.batch_size_per_device=64 \
#        trainer.max_epochs=10 \
#        trainer.devices=2 \
#        tokenizer="custom_tokenizer" \
#        embedder="resnet" \
#        optimizer="adamw" \
#        optimizer.fn.lr=0.001 \
#        logger.project="INTRONS_CL" \
#        wandb.api_key="765bad652bcb6ce569641fc334bcf0f0eea5b1fb" \

python -m scripts.constitutive_introns_finetuning \
    --csv_file "/data/ak5078/MSA_new/Constitutive_intron_sequences.csv" \
    --encoder_name "resnet" \
    --encoder_name_wb "Resnet" \
    --custom_tokenizer \
    --use_checkpoint \
    --simclr_ckpt "/home/ak5078/CL_Splicing_Regulation/checkpoints/introns_cl/ResNet1D/199/best-checkpoint.ckpt" \
    --classification_head_embedding_dimension 512 \
    --batch_size_per_device 16 \
    --global_batch_size 64 \
    --num_workers 8 \
    --learning_rate 0.0001 \
    --max_epochs 1 \
#python -m scripts.constitutive_introns_finetuning \
#    --csv_file "/data/ak5078/MSA_new/Constitutive_intron_sequences.csv" \
#    --custom_tokenizer False \
#    --encoder_name "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species" \
#    --encoder_name_wb "NTv2" \
#    --trained_ckpt "/home/ak5078/CL_Splicing_Regulation/checkpoints/introns_cl/NTv2/199/best-checkpoint.ckpt" \
#    --batch_size_per_device 16 \
#    --global_batch_size 64 \
#    --num_workers 8 \
#    --learning_rate 0.0001 \
#    --max_epochs 1 \
#    --devices 1 \

#python -m scripts.psi_lung_finetuning \
#    --csv_file "/data/ak5078/MSA_new/psi_Lung_intron_sequences.csv" \
#    --custom_tokenizer False \
#    --encoder_name "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species" \
#    --encoder_name_wb "NTv2" \
#    --regression_head_embedding_dimension 512 \
#    --trained_ckpt "/home/ak5078/CL_Splicing_Regulation/checkpoints/introns_cl/NTv2/199/best-checkpoint.ckpt" \
#    --batch_size_per_device 16 \
#    --global_batch_size 64 \
#    --num_workers 8 \
#    --learning_rate 0.0001 \
#    --max_epochs 1 \
#    --devices 1 \