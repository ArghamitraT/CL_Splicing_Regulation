import torch
from enformer_pytorch import from_pretrained
from .base_embedder import BaseEmbedder

class EnformerEmbedder(BaseEmbedder):
    def __init__(self, device):
        self.model = from_pretrained("EleutherAI/enformer-official-rough", use_tf_gamma=False, target_length=8).to(device)
        self.model.eval()
        self.device = device

    def embed(self, input_tensor):
        # return self.model(input_tensor.to(self.device))
        output = self.model(input_tensor.to(self.device))
        print(type(output))
        if isinstance(output, dict):
            print(output.keys())
        return output
