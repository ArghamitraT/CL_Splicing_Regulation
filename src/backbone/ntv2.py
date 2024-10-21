from transformers import AutoModelForMaskedLM
from src.backbone.base import BaseEncoder


class NTv2Embedder(BaseEncoder):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name, trust_remote_code=True).esm