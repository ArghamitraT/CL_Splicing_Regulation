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
    
    def forward(self, seql, seqr=None):
        
        # For MTSplice: seql and seqr already have shape: (B, 4, L)
        # For DilatedConv1D: seql has shape (B, 4, L) and seqr is None
        if seqr is not None:
            embedding = self.encoder(seql.float(), seqr.float())
        else:
            embedding = self.encoder(seql.float())
        z = self.projection_head(embedding)
        return z
    
    
def get_simclr_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = get_embedder(config).to(device)
    model = SimCLRModule(embedder=embedder, hidden_dim=config.model.hidden_dim, projection_dim=config.model.projection_dim)
    return model
