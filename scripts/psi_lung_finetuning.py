import os
import argparse
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoTokenizer
from src.datasets.lit import PsiLungIntronsDataModule
from src.model.psi_lung_introns import PsiLungIntronsRegressor
from src.tokenizers.custom import CustomTokenizer

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a model for intron classification.")

    parser.add_argument("--csv_file", type=str, required=True, help="Path to the CSV file containing intron sequences.")
    parser.add_argument("--custom_tokenizer", action="store_true", help="Use the custom tokenizer or not.")
    parser.add_argument("--encoder_name", type=str, required=True, help="Pre-trained encoder name or path.")
    parser.add_argument("--encoder_name_wb", type=str, required=True, help="Pre-trained encoder name for Weights and Biases run.")
    parser.add_argument("--regression_head_embedding_dimension", type=int, default=512, help="Number of workers for data loading.")
    parser.add_argument("--use_checkpoint", action="store_true", help="Use a pretrained checkpoint with SimCLR.")
    parser.add_argument("--simclr_ckpt", type=str, required=False, default=None, help="Path to the trained SimCLR checkpoint.")
    parser.add_argument("--batch_size_per_device", type=int, default=32, help="Batch size for each device.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for fine-tuning.")
    parser.add_argument("--max_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--logger_project", type=str, default="CONSTITUTIVE_INTRONS_FINETUNING", help="WandB project name.")
    parser.add_argument("--global_batch_size", type=int, default=None, help="Global batch size for training.")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use for training.")
    return parser.parse_args()

def calculate_accumulation_steps(global_batch_size, devices, batch_size_per_device):
    """
    Calculate gradient accumulation steps based on global batch size, devices, and batch size per device.
    """
    if global_batch_size is None:
        return 1  # Default accumulation steps
    total_device_batch_size = batch_size_per_device * devices
    if global_batch_size % total_device_batch_size != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by "
            f"(batch_size_per_device {batch_size_per_device} * devices {devices})."
        )
    return global_batch_size // total_device_batch_size

def main():
    # Parse arguments
    args = parse_arguments()
    
    tokenizer_map = {"nucleotide-transformer-v2": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species" }

    # Calculate gradient accumulation steps
    accumulation_steps = calculate_accumulation_steps(
        global_batch_size=args.global_batch_size,
        devices=args.devices,
        batch_size_per_device=args.batch_size_per_device
    )
    print(f"Using gradient accumulation steps: {accumulation_steps}")
    # Load tokenizer
    if args.custom_tokenizer:
        tokenizer = CustomTokenizer(model_max_length=201, padding="longest")
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_map[args.encoder_name], model_max_length=201, padding="longest")

    # Load data module
    data_module = PsiLungIntronsDataModule(
        csv_file=args.csv_file,
        tokenizer=tokenizer,
        padding_strategy='longest',
        batch_size=args.batch_size_per_device,
        num_workers=args.num_workers
    )
    data_module.setup()

    # Load model
    regressor = PsiLungIntronsRegressor(
        trained_simclr_ckpt=args.simclr_ckpt,
        encoder_name=args.encoder_name,
        classification_head_embedding_dimension=args.classification_head_embedding_dimension,
        lr=args.learning_rate,
        use_checkpoint=args.use_checkpoint,
    )

    # Initialize WandbLogger
    logger = WandbLogger(
        name=f"{args.encoder_name_wb}-lr{args.learning_rate}-bs{args.global_batch_size or args.batch_size_per_device * args.devices}",
        project=args.logger_project
    )

    # Initialize ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/psi_lung/{args.encoder_name_wb}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    # Create Trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        log_every_n_steps=1,
        val_check_interval = 120 * accumulation_steps,
        precision="16-mixed",
        gradient_clip_val=1.0,
        devices=args.devices,
        strategy="ddp" if args.devices > 1 else "auto",
        accumulate_grad_batches=accumulation_steps,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    # Train and test
    trainer.fit(regressor, data_module)
    trainer.test(regressor, data_module.test_dataloader())

if __name__ == "__main__":
    main()
