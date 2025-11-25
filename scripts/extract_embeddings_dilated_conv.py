#!/usr/bin/env python3
"""
Extract embeddings from DilatedConv1D pretraining checkpoint.

Loads a pretrained model and extracts embeddings for all samples in the dataset.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import lightning.pytorch as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.lit import LitModel
from src.model.simclr import get_simclr_model
from src.datasets.lit import ContrastiveIntronsDataModule
from src.custom_tokenizers.onehot_tokenizer import FastOneHotPreprocessor
from hydra.utils import instantiate


def load_config_from_checkpoint_dir(checkpoint_dir: Path) -> dict:
    """Load Hydra config from checkpoint directory."""
    config_file = checkpoint_dir / "configs" / "config.json"
    
    if config_file.exists():
        with open(config_file, "r") as f:
            return json.load(f)
    
    # Fallback to config.yaml if .json doesn't exist
    yaml_config = checkpoint_dir / "configs" / "config.yaml"
    if yaml_config.exists():
        return OmegaConf.to_container(OmegaConf.load(yaml_config))
    
    raise FileNotFoundError(f"Config not found in {checkpoint_dir}")


def extract_embeddings(
    checkpoint_path: str,
    config_dir: str,
    output_dir: str,
    split: str = "train",
    batch_size: int = 128,
    device: str = "cpu",
    max_batches: Optional[int] = None,
) -> None:
    """
    Extract embeddings from checkpoint using actual training data.
    
    Args:
        checkpoint_path: Path to .ckpt file
        config_dir: Path to checkpoint directory (contains configs/)
        output_dir: Directory to save embeddings
        split: Which split to extract from ('train', 'val', 'test')
        batch_size: Batch size for processing
        device: Device to use ('cpu' or 'cuda')
        max_batches: Maximum batches to process (None = all)
    """
    
    checkpoint_path = Path(checkpoint_path)
    config_dir = Path(config_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading config from {config_dir}...")
    config_dict = load_config_from_checkpoint_dir(config_dir)
    config = OmegaConf.create(config_dict)
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    print("Building model architecture...")
    model = get_simclr_model(config)
    
    print("Loading checkpoint weights...")
    # Extract state_dict from checkpoint and load into model
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    # Filter to only model weights (remove 'model.' prefix)
    model_state = {}
    for key, val in state_dict.items():
        if key.startswith("model."):
            new_key = key.replace("model.", "")
            model_state[new_key] = val
        else:
            model_state[key] = val
    
    model.load_state_dict(model_state, strict=False)
    model.eval()
    model = model.to(device)
    
    print(f"Initializing data module for split '{split}'...")
    data_module = ContrastiveIntronsDataModule(config)
    data_module.setup(stage="predict")
    
    # Get appropriate dataloader
    if split == "train":
        dataloader = data_module.train_dataloader()
    elif split == "val":
        dataloader = data_module.val_dataloader()
    elif split == "test":
        dataloader = data_module.test_dataloader()
    else:
        raise ValueError(f"Unknown split: {split}")
    
    print(f"Extracting embeddings from {split} set...")
    
    all_embeddings = []
    all_exon_ids = []
    all_exon_names = []
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            
            # Unpack batch
            *views, exon_ids, exon_names = batch
            
            # Extract embeddings for first view (they're all the same intronic region)
            view = views[0].float().to(device)
            
            # Get embedding from the model (without projection head)
            embeddings = model.encoder(view)  # Raw embeddings before projection
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_exon_ids.extend(exon_ids)
            all_exon_names.extend(exon_names)
            
            batch_count += 1
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                print(f"  Processed batch {batch_idx + 1}/{len(dataloader)}")
    
    # Concatenate all embeddings
    embeddings_array = np.concatenate(all_embeddings, axis=0)
    exon_ids_array = np.array(all_exon_ids)
    exon_names_array = np.array(all_exon_names)
    
    print(f"\nExtracted {embeddings_array.shape[0]} embeddings with shape {embeddings_array.shape}")
    
    # Save embeddings
    output_embeddings = output_dir / f"embeddings_{split}.npy"
    output_ids = output_dir / f"exon_ids_{split}.npy"
    output_names = output_dir / f"exon_names_{split}.npy"
    
    np.save(output_embeddings, embeddings_array)
    np.save(output_ids, exon_ids_array)
    np.save(output_names, exon_names_array)
    
    print(f"Saved embeddings to {output_embeddings}")
    print(f"Saved exon IDs to {output_ids}")
    print(f"Saved exon names to {output_names}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from DilatedConv1D pretraining checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (.ckpt)"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        required=True,
        help="Path to checkpoint directory (contains configs/)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/embeddings",
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Which split to extract embeddings from"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for extraction"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu or cuda)"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum batches to process (None = all)"
    )
    
    args = parser.parse_args()
    
    extract_embeddings(
        checkpoint_path=args.checkpoint,
        config_dir=args.config_dir,
        output_dir=args.output_dir,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
