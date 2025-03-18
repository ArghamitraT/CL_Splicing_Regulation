from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import wandb
import os
import sys
from lightning.pytorch.loggers import WandbLogger


def create_trainer(config: OmegaConf):
        # Initialize the logger
        wandb.login(key=config.wandb.api_key)

        #Check on the checkpoint directory
        if not os.path.exists(config.callbacks.model_checkpoint.dirpath):
                print(f"Creating directory {config.callbacks.model_checkpoint.dirpath}")
                os.makedirs(config.callbacks.model_checkpoint.dirpath)
                
        # Instantiate callbacks
        callbacks = []
        for cb_name, cb_conf in config.callbacks.items():
                callbacks.append(instantiate(cb_conf))

        # Explicitly initialize WandB Logger with increased timeout
        # wandb_logger = WandbLogger(
        #         settings=wandb.Settings(init_timeout=600)  # Set timeout to 10 minutes
        # )
        # Instantiate the trainer with the WandB logger
        # trainer = instantiate(config.trainer, callbacks=callbacks, logger=wandb_logger)
        
        logger = instantiate(config.logger)     
        # Instantiate the trainer with the WandB logger
        trainer = instantiate(config.trainer, callbacks=callbacks, logger=logger)
        
        return trainer

        """
        # (AT)
        logger = instantiate(config.logger)      
        # Instantiate the trainer
        trainer = instantiate(config.trainer, callbacks=callbacks, logger=logger)
        return trainer
        """
        