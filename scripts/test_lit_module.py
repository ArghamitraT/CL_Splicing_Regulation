import hydra
from omegaconf import OmegaConf
from src.utils.config import process_config,print_config
import torch
import numpy as np
import random
from src.model.lit import create_lit_model
from src.trainer.utils import create_trainer
from lightly import loss
from src.datasets.lit import DummyDataModule
from transformers import AutoTokenizer


@hydra.main(version_base=None,config_path="../configs", config_name="config.yaml")
def main(config:OmegaConf):

    OmegaConf.register_new_resolver('eval', eval)
    OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
    OmegaConf.register_new_resolver('min', lambda x, y: min([x, y]))
    OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)

    print_config(config, resolve=True)
    
    model= create_lit_model(config)
    
    trainer = create_trainer(config)

    
    data_module = DummyDataModule(config.dataset.seq_len, config.dataset.num_pairs, config.dataset.train_batch_size, tokenizer_name=config.backbone.name_or_path)
    data_module.prepare_data()
    data_module.setup()
    
    trainer.fit(model,data_module.train_dataloader(),data_module.val_dataloader())
    
    
if __name__ == "__main__":
    main()