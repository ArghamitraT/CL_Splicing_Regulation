import hydra
import torch
from omegaconf import OmegaConf
from src.utils.config import process_config,print_config
import numpy as np



@hydra.main(version_base=None,config_path="../configs", config_name="config.yaml")
def main(config:OmegaConf):

    OmegaConf.register_new_resolver('eval', eval)
    OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
    OmegaConf.register_new_resolver('min', lambda x, y: min([x, y]))
    OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)

    print_config(config, resolve=True)
if __name__ == "__main__":
    main()