from transformers import AutoModelForMaskedLM
from src.backbone.base import BaseEncoder

class NTv2Embedder(BaseEncoder):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.name_or_path, trust_remote_code=True).esm