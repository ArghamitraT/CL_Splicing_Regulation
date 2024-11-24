import torch
import torch.nn as nn
from src.embedder.utils import get_embedder
from lightly.models.modules import heads

class SimCLRModule(nn.Module):
    def __init__(self, embedder, hidden_dim=512, projection_dim=128):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        
        self.encoder = embedder
        self.projection_head = heads.SimCLRProjectionHead(
            input_dim=embedder.get_last_embedding_dimension(),
            hidden_dim=self.hidden_dim ,
            output_dim=self.projection_dim,
        )

    def forward(self, x):
        features = self.encoder(x)
        embedding = features.mean(dim=1)
        z = self.projection_head(embedding)
        return z
    
    
def get_simclr_model(config, state_dict=None):
    embedder = get_embedder(config)
    model = SimCLRModule(embedder=embedder, hidden_dim=config.model.hidden_dim, projection_dim=config.model.projection_dim)
    if state_dict is not None:
        # Remove the "model." prefix from the keys
        adjusted_state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(adjusted_state_dict)

    return model