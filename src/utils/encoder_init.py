import torch
from hydra.utils import instantiate
from src.model.simclr import get_simclr_model
from src.model import MTSpliceBCE
import torch.nn as nn
from pathlib import Path
import os

def init_weights_he_normal(module):
    """
    Applies He normal (Kaiming normal) initialization to
    Linear and Convolutional layers.
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def find_latest_pretrain_checkpoint(root_path, embedder_name):
    """
    Auto-discover the latest pretrain checkpoint for the given embedder.
    
    Args:
        root_path: Root path to CLADES
        embedder_name: Name of embedder (e.g., 'MTSplice', 'DilatedConv1D')
    
    Returns:
        Path to checkpoint file or None if not found
    """
    output_dir = Path(root_path) / "output"
    
    # Find all pretrain directories
    pretrain_dirs = sorted(
        [d for d in output_dir.glob("pretrain_*") if d.is_dir()],
        reverse=True  # Most recent first
    )
    
    for pretrain_dir in pretrain_dirs:
        ckpt_path = pretrain_dir / "checkpoints" / "introns_cl" / embedder_name / "400" / "best-checkpoint.ckpt"
        if ckpt_path.exists():
            print(f"✨ Found pretrain checkpoint for {embedder_name}: {pretrain_dir.name}")
            return str(ckpt_path)
    
    return None

def load_encoder(config, root_path, result_dir=None):

    simclr_model = get_simclr_model(config)

    if config.aux_models.warm_start:
        
        # If result_dir not provided or checkpoint doesn't exist, auto-discover
        if result_dir:
            simclr_ckpt = f"{root_path}/output/{result_dir}/checkpoints/introns_cl/{config.embedder._name_}/400/best-checkpoint.ckpt"
        else:
            simclr_ckpt = None
        
        # Check if the configured checkpoint exists
        if result_dir and not os.path.exists(simclr_ckpt):
            print(f"⚠️ Checkpoint not found at {simclr_ckpt}, attempting auto-discovery...")
            simclr_ckpt = find_latest_pretrain_checkpoint(root_path, config.embedder._name_)
        elif not simclr_ckpt:
            # Auto-discover if no result_dir provided
            simclr_ckpt = find_latest_pretrain_checkpoint(root_path, config.embedder._name_)
        
        if not simclr_ckpt:
            raise FileNotFoundError(
                f"No pretrain checkpoint found for embedder '{config.embedder._name_}'. "
                f"Please run pretraining first or check the config."
            )
        
        if torch.cuda.is_available():
            ckpt = torch.load(simclr_ckpt, weights_only=False)  # load normally on GPU
            device = torch.device("cuda")
        else:
            ckpt = torch.load(simclr_ckpt, map_location=torch.device("cpu"), weights_only=False)  # fallback to CPU
            device = torch.device("cpu")
        
        state_dict = ckpt["state_dict"]
        cleaned_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        missing, unexpected = simclr_model.load_state_dict(cleaned_state_dict, strict=False)
        print(f"✅ Loaded pre-trained weights for encoder: {config.embedder._name_}")

    elif config.aux_models.He_Normalinitial:
        print(f"⚠️ 'warm_start' is False. Applying He Normal initialization to encoder.")
        simclr_model.encoder.apply(init_weights_he_normal)
            
    return simclr_model.encoder


def initialize_encoders_and_model(config, root_path):
    
    # Try to use configured weights, but fall back to auto-discovery
    result_dir = config.aux_models.get("mtsplice_weights", None)
    
    encoder = load_encoder(config, root_path, result_dir)
    return MTSpliceBCE.MTSpliceBCE(encoder, config)
    

