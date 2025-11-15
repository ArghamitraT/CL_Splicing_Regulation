import torch
from hydra.utils import instantiate
from src.model.simclr import get_simclr_model
from src.model import MTSpliceBCE
import torch.nn as nn



# --- NEW ---
# 1. Define the He Normal initialization function
def init_weights_he_normal(module):
    """
    Applies He normal (Kaiming normal) initialization to
    Linear and Convolutional layers.
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        # Apply Kaiming normal initialization to the weights
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        
        # Initialize biases to zero (a common practice)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
# --- END NEW ---

def load_encoder(config, root_path, result_dir):
    
    # Get the SimCLR model (which contains the encoder + projection head)
    simclr_model = get_simclr_model(config)

    if config.aux_models.warm_start:
        

        simclr_ckpt = f"{root_path}/output/{result_dir}/checkpoints/introns_cl/{config.embedder._name_}/400/best-checkpoint.ckpt"
     
        if torch.cuda.is_available():
            ckpt = torch.load(simclr_ckpt)  # load normally on GPU
            device = torch.device("cuda")
        else:
            ckpt = torch.load(simclr_ckpt, map_location=torch.device("cpu"))  # fallback to CPU
            device = torch.device("cpu")
        
        state_dict = ckpt["state_dict"]
        cleaned_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        missing, unexpected = simclr_model.load_state_dict(cleaned_state_dict, strict=False)
        print(f"✅ Loaded pre-trained weights for encoder from {result_dir}")
        # print()
    
    # --- NEW ---
    # 2. Add 'else' block to initialize encoder if not warm-starting
    elif config.aux_models.He_Normalinitial:
        # Not pre-initializing, so apply He Normal initialization to the encoder
        print(f"⚠️ 'warm_start' is False. Applying He Normal initialization to encoder.")
        simclr_model.encoder.apply(init_weights_he_normal)
    # --- END NEW ---
            
    return simclr_model.encoder


def initialize_encoders_and_model(config, root_path):
    
    result_dirs = {
        "mtsplice_weights": config.aux_models["mtsplice_weights"]
    }

    
    encoder = load_encoder(config, root_path, result_dirs["mtsplice_weights"])
    return MTSpliceBCE.MTSpliceBCE(encoder, config)
    
