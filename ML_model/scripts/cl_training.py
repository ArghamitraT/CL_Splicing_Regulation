import sys
import os

# Add the parent directory (main) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hydra
from omegaconf import OmegaConf
from src.utils.config import  print_config
import torch
from src.model.lit import create_lit_model
from src.trainer.utils import create_trainer
from src.datasets.lit import ContrastiveIntronsDataModule

# os.environ['WANDB_INIT_TIMEOUT'] = '600'
def get_optimal_num_workers():
    num_cpus = os.cpu_count()
    num_gpus = torch.cuda.device_count()
    return min(num_cpus // max(1, num_gpus), 16)

@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def main(config: OmegaConf):

    # Register Hydra resolvers
    OmegaConf.register_new_resolver('eval', eval)
    OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
    OmegaConf.register_new_resolver('min', lambda x, y: min([x, y]))
    OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
    OmegaConf.register_new_resolver('optimal_workers', lambda: get_optimal_num_workers())

    # Print and process configuration
    print_config(config, resolve=True)


    # Initialize the IntronsDataModule with dataset-specific configs
    data_module = ContrastiveIntronsDataModule(config
    )
    data_module.prepare_data()
    data_module.setup()
    
    tokenizer = data_module.tokenizer
    
    lit_model = create_lit_model(config)
    
    trainer = create_trainer(config)
    
    trainer.fit(lit_model, data_module.train_dataloader(), data_module.val_dataloader())
if __name__ == "__main__":
    main()
