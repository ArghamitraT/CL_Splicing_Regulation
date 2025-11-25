"""
1D Dilated Convolutional Neural Network Embedder

Based on IsoCLR framework, implements dilated convolutions for efficient
long-range context capture in sequence data with minimal parameter growth.

Key features:
- Dilated convolutions with exponentially growing receptive field
- Residual connections for better gradient flow
- Configurable depth, width, and dilation strategy
- Batch normalization and activation functions
- Suitable for sequence lengths 100-500+ bp
- Compatible with CLADES framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from src.embedder.base import BaseEmbedder


class DilatedConvBlock(nn.Module):
    """
    A single dilated convolution block with residual connection.
    
    Architecture:
        Input → Conv1d(dilation) → BatchNorm → ReLU → Dropout → Output + Residual
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolution kernel
        dilation: Dilation factor (> 1 increases receptive field)
        dropout: Dropout probability (0 = no dropout)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False, 
        )
        self.batch_norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.residual_conv = None
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, in_channels, length)
        
        Returns:
            Output tensor of shape (batch, out_channels, length)
        """
        residual = x

        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.activation(out)
        out = self.dropout(out)

        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        
        out = out + residual
        return out


class DilatedConv1DNet(nn.Module):
    """
    Dilated 1D Convolutional Network Encoder.
    
    Uses exponentially growing dilations to efficiently capture long-range
    dependencies while maintaining a small parameter count.
    
    Receptive field grows as: RF = 1 + 2 * sum(kernel_size * dilation)
    
    Args:
        input_dim: Input feature dimension (e.g., 11 for one-hot DNA)
        embedding_dim: Output embedding dimension
        num_layers: Number of dilated conv blocks
        base_channels: Base number of channels
        kernel_size: Kernel size for all convolutions
        dilation_strategy: 'exponential' (1,2,4,8...) or 'linear' (1,2,3,4...)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int = 11,
        embedding_dim: int = 768,
        num_layers: int = 4,
        base_channels: int = 64,
        kernel_size: int = 3,
        dilation_strategy: str = "exponential",
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.initial_proj = nn.Sequential(
            nn.Conv1d(input_dim, base_channels, kernel_size=1),
            nn.BatchNorm1d(base_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.ModuleList()
        dilations = self._get_dilations(dilation_strategy, num_layers)
        
        for dilation in dilations:
            block = DilatedConvBlock(
                in_channels=base_channels,
                out_channels=base_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            )
            self.blocks.append(block)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.output_proj = nn.Sequential(
            nn.Linear(base_channels, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
    
    @staticmethod
    def _get_dilations(strategy: str, num_layers: int) -> List[int]:
        """Generate dilation factors based on strategy."""
        if strategy == "exponential":
            return [2 ** i for i in range(num_layers)]
        elif strategy == "linear":
            return list(range(1, num_layers + 1))
        else:
            raise ValueError(f"Unknown dilation_strategy: {strategy}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim) or (batch, input_dim, seq_len)
        
        Returns:
            Embedding tensor of shape (batch, embedding_dim)
        """
        if x.dim() == 3 and x.shape[2] == self.input_dim:
            x = x.transpose(1, 2)  

        x = self.initial_proj(x)

        # Apply dilated conv blocks
        for block in self.blocks:
            x = block(x)

        # Global average pooling
        x = self.pool(x) 
        x = x.squeeze(-1) 
        
        # Output projection
        x = self.output_proj(x) 
        
        return x


class DilatedConv1DWithMaxPooling(nn.Module):
    """
    Dilated 1D Conv Network with optional max-pooling stages.
    
    Similar to DilatedConv1DNet but includes max-pooling layers to gradually
    reduce sequence length while increasing receptive field more aggressively.
    
    Args:
        input_dim: Input feature dimension
        embedding_dim: Output embedding dimension
        num_layers: Number of dilated conv blocks
        base_channels: Base number of channels
        kernel_size: Kernel size
        pool_kernel_size: Max pooling kernel size (if None, no pooling)
        dilation_strategy: 'exponential' or 'linear'
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int = 11,
        embedding_dim: int = 768,
        num_layers: int = 4,
        base_channels: int = 64,
        kernel_size: int = 3,
        pool_kernel_size: Optional[int] = 2,
        dilation_strategy: str = "exponential",
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        self.initial_proj = nn.Sequential(
            nn.Conv1d(input_dim, base_channels, kernel_size=1),
            nn.BatchNorm1d(base_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        dilations = self._get_dilations(dilation_strategy, num_layers)
        
        for i, dilation in enumerate(dilations):
            block = DilatedConvBlock(
                in_channels=base_channels,
                out_channels=base_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            )
            self.blocks.append(block)
            
            # Add max pooling between blocks (except last)
            if pool_kernel_size is not None and i < num_layers - 1:
                self.pools.append(
                    nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size)
                )
            else:
                self.pools.append(nn.Identity())

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.output_proj = nn.Sequential(
            nn.Linear(base_channels, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
    
    @staticmethod
    def _get_dilations(strategy: str, num_layers: int) -> List[int]:
        if strategy == "exponential":
            return [2 ** i for i in range(num_layers)]
        elif strategy == "linear":
            return list(range(1, num_layers + 1))
        else:
            raise ValueError(f"Unknown dilation_strategy: {strategy}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[2] == self.input_dim:
            x = x.transpose(1, 2)
        
        x = self.initial_proj(x)

        for block, pool in zip(self.blocks, self.pools):
            x = block(x)
            x = pool(x)

        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.output_proj(x)
        
        return x


class DilatedConvEmbedder(BaseEmbedder):
    """
    Main embedder class compatible with CLADES framework.
    
    Wraps DilatedConv1DNet for use with Hydra config system.
    
    Config usage (in configs/embedder/dilated_conv_1d.yaml):
        _target_: src.embedder.dilated_conv_1d.DilatedConvEmbedder
        embedding_dim: 768
        input_dim: 4  # Matches FastOneHotPreprocessor output (A, C, G, T channels)
        num_layers: 4
        base_channels: 64
        kernel_size: 3
        dilation_strategy: exponential
        dropout: 0.1
        use_max_pooling: false
    """
    
    def __init__(
        self,
        input_dim: int = 11,
        embedding_dim: int = 768,
        seq_len: int = 200,
        num_layers: int = 4,
        base_channels: int = 64,
        kernel_size: int = 3,
        dilation_strategy: str = "exponential",
        dropout: float = 0.1,
        use_max_pooling: bool = False,
        pool_kernel_size: Optional[int] = 2,
        **kwargs 
    ):
        super().__init__(name_or_path="DilatedConv1D", bp_per_token=kwargs.get('bp_per_token', None))
        
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.input_dim = input_dim

        if use_max_pooling:
            self.encoder = DilatedConv1DWithMaxPooling(
                input_dim=input_dim,
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                base_channels=base_channels,
                kernel_size=kernel_size,
                pool_kernel_size=pool_kernel_size,
                dilation_strategy=dilation_strategy,
                dropout=dropout,
            )
        else:
            self.encoder = DilatedConv1DNet(
                input_dim=input_dim,
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                base_channels=base_channels,
                kernel_size=kernel_size,
                dilation_strategy=dilation_strategy,
                dropout=dropout,
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, input_dim) or (batch, input_dim, seq_len)
        
        Returns:
            Embeddings of shape (batch, embedding_dim)
        """
        return self.encoder(x)
    
    def get_last_embedding_dimension(self) -> int:
        """
        Returns the output embedding dimension for framework compatibility.
        
        Returns:
            int: The embedding dimension
        """
        return self.embedding_dim
