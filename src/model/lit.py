import torch
import torch.nn as nn
import lightning.pytorch as pl
import hydra
from hydra.utils import instantiate
from src.model.simclr import get_simclr_model

class LitModel(pl.LightningModule):
    """
    Modular PyTorch Lightning model adaptable for various tasks.

    Args:
        model (nn.Module): The model (simCLR) that will be trained through supervised learning.
        optimizer_class (torch.optim.Optimizer): The optimizer class to use.
        learning_rate (float): The learning rate for the optimizer.
        **kwargs: Additional hyperparameters to save.
    """
    def __init__(
        self,
        model,
        config,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.config = config

        # Initialize Loss from config
        self.loss_fn = instantiate(config.loss)
    
    def setup(self, stage=None):
        """
        Setup function for model training. This function is called at the beginning of training
        and validation, and it allows the model to prepare its environment for the given stage.

        Args:
            stage (str): Either 'fit' for training or 'validate' for validation.
        """
        if stage == 'fit' or stage is None:
            # This is where you might handle any logic related to setting up the training environment,
            # such as initializing specific data, setting up tasks, or preloading model weights.
            print(f"Setting up training for {self.config.task._name_}")
            
        if stage == 'validate' or stage is None:
            # Setup for validation
            print(f"Setting up validation for {self.config.task._name_}")

    def forward(self, *inputs):
        return self.model.forward(*inputs)

    def training_step(self, batch, batch_idx):
        view0,view1 = batch
        # Forward pass
        z0 = self.forward(view0)
        z1 = self.forward(view1)
        
        loss = self.loss_fn(z0, z1)
        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        view0,view1 = batch
        # Forward pass
        z0 = self.forward(view0)
        z1 = self.forward(view1)
        
        loss = self.loss_fn(z0, z1)
        self.log('val_loss', loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer = hydra.utils.instantiate(self.config.optimizer,params=params)
        return optimizer
    
def create_lit_model(config):

    simclr_model = get_simclr_model(config)

    # Instantiate LitModel
    lit_model = LitModel(
        model=simclr_model,
        config=config,
    )
    return lit_model
