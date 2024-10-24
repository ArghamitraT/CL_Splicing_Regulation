import hydra
from omegaconf import OmegaConf
from src.utils.config import process_config, print_config
import torch
import random
from src.model.lit import create_lit_model
from src.trainer.utils import create_trainer
from src.datasets.lit import IntronsDataModule
from transformers import AutoTokenizer


@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def main(config: OmegaConf):

    # Register Hydra resolvers
    OmegaConf.register_new_resolver('eval', eval)
    OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
    OmegaConf.register_new_resolver('min', lambda x, y: min([x, y]))
    OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)

    # Print and process configuration
    print_config(config, resolve=True)


    # Initialize the IntronsDataModule with dataset-specific configs
    data_module = IntronsDataModule(config
    )
    data_module.prepare_data()
    data_module.setup()

    loader = data_module.train_dataloader()
    for batch in loader:
        print(f" Batch loaded with 2 views of shape: {batch[0].shape}")
        print(f"First sequence in view 1: {batch[0][0]}")
        print(f"First sequence in view 2: {batch[1][0]}")
        break


if __name__ == "__main__":
    main()
