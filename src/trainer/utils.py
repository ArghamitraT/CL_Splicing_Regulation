from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import wandb
import os
import sys
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint # <-- Import ModelCheckpoint


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
        
        # --- MINIMAL FIX STARTS HERE ---
        # Check if checkpointing is disabled in the trainer config
        if hasattr(config.trainer, 'enable_checkpointing') and not config.trainer.enable_checkpointing:
            print("Checkpointing disabled by config.trainer.enable_checkpointing=False. Removing ModelCheckpoint from callbacks.")
            # Filter out any ModelCheckpoint instances from the list
            callbacks = [cb for cb in callbacks if not isinstance(cb, ModelCheckpoint)]
        # --- MINIMAL FIX ENDS HERE ---

        if config.logger._target_ == "lightning.pytorch.loggers.WandbLogger":
                wandb_logger = WandbLogger(
                        name=config.logger.name,
                        project=config.logger.project,
                        group=config.logger.group,
                        save_dir=config.logger.save_dir,
                        log_model=config.logger.log_model,
                        config=OmegaConf.to_container(config, resolve=True),  # ✅ this logs the config
                        settings=wandb.Settings(init_timeout=600)
                )
                
                logger = wandb_logger
        else:
                logger = instantiate(config.logger)
        trainer = instantiate(config.trainer, callbacks=callbacks, logger=logger)
           
        return trainer
