from borzoi_pytorch import Borzoi
from src.embedder.base import BaseEmbedder


class BorzoiEmbedder(BaseEmbedder):
    
    def __init__(self, seq_len: int, **kwargs):
        super().__init__(name_or_path="BorzoiEmbedder", bp_per_token=kwargs.get("bp_per_token", None))
        self.backbone = Borzoi.from_pretrained('johahi/borzoi-replicate-0')
        self.seq_len = seq_len
    
    def forward(self, input_ids, **kwargs):
        print(f"Before fwd: {input_ids.shape}")
        self.backbone(input_ids, **kwargs)
