"""
Model utilities for CLADES framework.
Consolidates model initialization and encoder handling logic.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


def get_encoder_output_dimension(encoder: nn.Module, 
                                seq_length: int = 400, 
                                test_input_shape: Optional[Tuple] = None) -> int:
    """
    Infer encoder output (embedding) dimension.
    
    Eliminates duplicate encoder dimension detection code across models.
    Uses get_last_embedding_dimension() if available, otherwise runs inference.
    
    Args:
        encoder: Encoder model
        seq_length: Sequence length for test input
        test_input_shape: Optional explicit test input shape
    
    Returns:
        Output embedding dimension
    
    Raises:
        RuntimeError: If dimension cannot be inferred
    """
    # First try: check if encoder has built-in method
    if hasattr(encoder, 'get_last_embedding_dimension'):
        try:
            dim = encoder.get_last_embedding_dimension()
            if dim is not None:
                return int(dim)
        except Exception:
            pass
    
    # Second try: run inference to determine dimension
    try:
        encoder_device = next(encoder.parameters()).device
        
        # Determine test input shape
        if test_input_shape is not None:
            test_input = torch.randn(*test_input_shape, device=encoder_device)
        else:
            # Default: assume DNA sequence (4 channels) for one-hot encoding
            test_input = torch.randn(1, 4, seq_length, device=encoder_device)
        
        with torch.no_grad():
            output = encoder(test_input)
            
            if isinstance(output, tuple):
                output = output[0]
            
            # Output should be [batch, embedding_dim]
            if len(output.shape) == 2:
                return output.shape[1]
            else:
                return output.numel() // output.shape[0]
    
    except Exception as e:
        raise RuntimeError(
            f"Could not infer encoder output dimension. "
            f"Either add get_last_embedding_dimension() method to encoder "
            f"or ensure test input shape is correct. Error: {e}"
        )


def compute_encoder_embeddings(encoder: nn.Module, 
                               *inputs: torch.Tensor) -> torch.Tensor:
    """
    Compute embeddings from encoder with flexible input handling.
    
    Supports single input or multiple inputs (e.g., for MTSplice with left/right sequences).
    
    Args:
        encoder: Encoder model
        *inputs: One or more input tensors
    
    Returns:
        Embedding tensor [batch_size, embedding_dim]
    """
    if len(inputs) == 1:
        return encoder(inputs[0])
    elif len(inputs) == 2:
        return encoder(inputs[0], inputs[1])
    else:
        return encoder(*inputs)


def initialize_model_weights(model: nn.Module, weight_init_fn=None) -> None:
    """
    Initialize model weights using specified initialization function.
    
    Consolidates weight initialization patterns.
    
    Args:
        model: Model to initialize
        weight_init_fn: Function to apply to each module (e.g., init_weights_he_normal)
    """
    if weight_init_fn is not None:
        model.apply(weight_init_fn)


def freeze_encoder_weights(encoder: nn.Module) -> None:
    """
    Freeze all encoder weights (set requires_grad=False).
    
    Args:
        encoder: Encoder module to freeze
    """
    for param in encoder.parameters():
        param.requires_grad = False


def unfreeze_encoder_weights(encoder: nn.Module) -> None:
    """
    Unfreeze all encoder weights (set requires_grad=True).
    
    Args:
        encoder: Encoder module to unfreeze
    """
    for param in encoder.parameters():
        param.requires_grad = True


def get_trainable_parameter_count(model: nn.Module) -> int:
    """
    Count trainable parameters in model.
    
    Args:
        model: Model to count parameters for
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_total_parameter_count(model: nn.Module) -> int:
    """
    Count total parameters in model (trainable + frozen).
    
    Args:
        model: Model to count parameters for
    
    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: nn.Module) -> float:
    """
    Estimate model size in MB.
    
    Args:
        model: Model to estimate size for
    
    Returns:
        Model size in megabytes
    """
    param_size = get_total_parameter_count(model) * 4  # 4 bytes per float32
    return param_size / (1024 * 1024)

