import torch
import torch.nn as nn
from borzoi_pytorch import Borzoi
from src.embedder.base import BaseEmbedder


class BorzoiEmbedder(BaseEmbedder):
    
    def __init__(self, seq_len, **kwargs):
        super().__init__(name_or_path="BorzoiEmbedder", bp_per_token=kwargs.get("bp_per_token", None))
        self.backbone = Borzoi.from_pretrained('johahi/borzoi-replicate-0')
        
        # Skip cropping layer for variable length
        self.backbone.crop = nn.Identity()
        
        # Interpolate to handle non-power of 2 length inputs
        self.backbone.get_embs_after_crop = self.get_embs
        self.seq_len = seq_len
    
    def forward(self, input_ids, **kwargs):
        z = self.backbone(input_ids.half(), **kwargs)
        z = z.mean(dim=-1)
        return z
    
    def align_and_add(self, x, y):
        """
        Aligns tensor y to x along tensor length. Then adds
        """
        if x.shape[-1] != y.shape[-1]:
            y = torch.nn.functional.interpolate(y, size=x.shape[-1], mode="nearest")
        return x + y
    
    def get_embs(self, x):
        """
        Performs the forward pass of the model until right before the final conv layers, and includes a cropping layer.

        Args:
            x (torch.Tensor): Input DNA sequence tensor of shape (N, 4, L).

        Returns:
             torch.Tensor: Output of the model up to the cropping layer with shape (N, dim, crop_length)
        """
        x = self.backbone.conv_dna(x)
        x_unet0 = self.backbone.res_tower(x)
        x_unet1 = self.backbone.unet1(x_unet0)
        x = self.backbone._max_pool(x_unet1)
        x_unet1 = self.backbone.horizontal_conv1(x_unet1)
        x_unet0 = self.backbone.horizontal_conv0(x_unet0)
        x = self.backbone.transformer(x.permute(0,2,1))
        x = x.permute(0,2,1)
        x = self.backbone.upsampling_unet1(x)
        
        # Handle mismatching bin sizes (seq len not power of 2)
        # x += x_unet1
        x = self.align_and_add(x, x_unet1)

        x = self.backbone.separable1(x)
        x = self.backbone.upsampling_unet0(x)
        
        # Handle mismatching bin sizes (seq len not power of 2)
        # x += x_unet0
        x = self.align_and_add(x, x_unet0)

        x = self.backbone.separable0(x)
        x = self.backbone.crop(x.permute(0,2,1))
        return x.permute(0,2,1)
    
    def get_last_embedding_dimension(self) -> int:
        """
        Function to get the last embedding dimension of a PyTorch model by passing
        a random tensor through the model and inspecting the output shape.
        This is done with gradients disabled and always on GPU.

        Args:
            model (nn.Module): The PyTorch model instance.

        Returns:
            int: The last embedding dimension (i.e., the last dimension of the output tensor).
        """

        # Try to determine the input shape based on the first layer of the model
        input_shape = (4, self.seq_len)

        DEVICE = next(self.backbone.parameters()).device
        random_input = torch.randint(low=0, high=2, size=(10, *input_shape)).float().to(DEVICE)
        print(f"Test input: {random_input.shape}")

        with torch.no_grad():
            output = self.backbone(random_input)

        output = output.mean(dim=-1)
        last_embedding_dimension = output.shape[-1]
        print(f"Found a last embedding dimension of {last_embedding_dimension}")
        return last_embedding_dimension
