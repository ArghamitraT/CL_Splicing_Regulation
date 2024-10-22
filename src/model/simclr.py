import torch
import torch.nn as nn
from src.backbone.utils import get_backbone
from lightly.models.modules import heads

class SimCLRModule(nn.Module):
    def __init__(self, backbone, hidden_dim=512, projection_dim=128):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        
        self.encoder = backbone
        self.projection_head = heads.SimCLRProjectionHead(
            input_dim=backbone.get_backbone_last_embedding_dimension(),
            hidden_dim=self.hidden_dim ,
            output_dim=self.projection_dim,
        )

    def forward(self, x):
        features = self.encoder(x)
        embedding = features.mean(dim=1)
        z = self.projection_head(embedding)
        return z
    
    
def get_simclr_model(config):
    print(f"Instantiating the Backbone")
    backbone = get_backbone(config)
    model = SimCLRModule(backbone, hidden_dim=config.model.hidden_dim, projection_dim=config.model.projection_dim)
    return model