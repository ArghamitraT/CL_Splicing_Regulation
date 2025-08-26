import torch
from borzoi_pytorch import Borzoi
from src.embedder.base import BaseEmbedder


class BorzoiEmbedder(BaseEmbedder):
    
    def __init__(self, seq_len, **kwargs):
        super().__init__(name_or_path="BorzoiEmbedder", bp_per_token=kwargs.get("bp_per_token", None))
        self.backbone = Borzoi.from_pretrained('johahi/borzoi-replicate-0')
        self.seq_len = seq_len
    
    def forward(self, input_ids, **kwargs):
        return self.backbone(input_ids, **kwargs)
    
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

        last_embedding_dimension = output.shape[-1]
        print(f"Found a last embedding dimension of {last_embedding_dimension}")
        return last_embedding_dimension
