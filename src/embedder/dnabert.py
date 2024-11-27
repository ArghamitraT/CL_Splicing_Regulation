import torch
import torch.nn as nn
from transformers import AutoModel
from src.embedder.base import BaseEmbedder
from transformers.models.bert.configuration_bert import BertConfig

class DNABERT2Embedder(BaseEmbedder):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone = self.initialize_dnabert2()
    
    def initialize_dnabert2(self):
        config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        backbone = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)
        backbone.pooler = None
        return backbone
    
    def forward(self, input_ids, **kwargs):
        """Extract embeddings from the input IDs"""
        return self.backbone(input_ids, **kwargs)[0]
    
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
        DEVICE = self.backbone.device
        input_shape = (64,)
        random_input = torch.randint(low=0, high=2, size=(10, *input_shape)).to(DEVICE)
        
        # Pass the tensor through the model with no gradients
        with torch.no_grad():
            print(f"input shape: {random_input.shape}")
            output = self(random_input)
            print(f"Output of the model of shape: {output.shape}")
            
            #Expted output of shape (batch_size, seq_len, embedding_dim)
            
        # Get the shape of the output tensor
        last_embedding_dimension = output.shape[-1]
        # Return the last dimension of the output tensor
        print(f"Found a last embedding dimension of {last_embedding_dimension}")
        return last_embedding_dimension
