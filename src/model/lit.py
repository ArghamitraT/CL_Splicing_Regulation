import torch
import torch.nn as nn
import lightning.pytorch as pl
import hydra
from hydra.utils import instantiate
from src.model.simclr import get_simclr_model
from src.utils.training_utils import log_epoch_timing, log_gpu_memory_stats, clear_gpu_cache
import time

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
        self.epoch_start_time = None  # Initialize time tracker

    def on_train_epoch_start(self):
        """Called at the start of each training epoch to record time."""
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        """Called at the end of each training epoch to log the time taken."""
        log_epoch_timing(self.log, self.current_epoch, self.epoch_start_time)
        log_gpu_memory_stats(self.log, self.current_epoch)
        clear_gpu_cache()

    
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

        import time
        start = time.time()
        *views, exon_ids, exon_names = batch
        

        # Forward pass for all views
        start_fwd = time.time()
      
        if self.config.embedder._name_ == 'MTSplice':
            z_views = [self.forward(seql, seqr) for (seql, seqr) in views]
        else:
            z_views = [self.forward(view) for view in views]
        

        # Loss computation and safety checks
        loss_func_name = self.loss_fn.__class__.__name__
        if 'SupConLoss' in loss_func_name:
            features = torch.stack(z_views, dim=1)  # [batch, n_views, emb_dim]
            if loss_func_name == 'weightedSupConLoss':
                division = 'train'
                loss = self.loss_fn(features, exon_names, division)
            else:
                loss = self.loss_fn(features)
        else:
            if len(z_views) != 2:
                raise ValueError(
                    f"{self.loss_fn.__class__.__name__} only supports 2 views per sample "
                    f"(got {len(z_views)} views). If you want more, use SupConLoss."
                )
            loss = self.loss_fn(z_views[0], z_views[1])

        self.log(
            'train_loss',
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=z_views[0].shape[0]
        )
        step_time = time.time() - start
        
        # --- GPU memory logging every N batches ---
        # if batch_idx % 50 == 0 and torch.cuda.is_available():
        #     torch.cuda.synchronize()
        #     allocated = torch.cuda.memory_allocated(0) / (1024**3)
        #     reserved  = torch.cuda.memory_reserved(0) / (1024**3)
        #     peak_alloc = torch.cuda.max_memory_allocated(0) / (1024**3)
        #     peak_reserved = torch.cuda.max_memory_reserved(0) / (1024**3)
        #     print(f"[Batch {batch_idx}] "
        #         f"Allocated: {allocated:.2f} GiB | "
        #         f"Reserved: {reserved:.2f} GiB | "
        #         f"PeakAlloc: {peak_alloc:.2f} GiB | "
        #         f"PeakReserved: {peak_reserved:.2f} GiB")

        return loss



    def validation_step(self, batch, batch_idx):
        import time
        start = time.time()

        # Unpack all views and exon_names from batch
        *views, exon_ids, exon_names = batch

        # Forward pass for all views
        start_fwd = time.time()
        if self.config.embedder._name_ == 'MTSplice':
            # MTSpliceEncoder expects a tuple of sequences
            z_views = [self.forward(seql, seqr) for (seql, seqr) in views]
        else:
            z_views = [self.forward(view) for view in views]
        
        # Loss computation and safety checks
        loss_func_name = self.loss_fn.__class__.__name__
        if 'SupConLoss' in loss_func_name:
            features = torch.stack(z_views, dim=1)  # [batch, n_views, emb_dim]
            if loss_func_name == 'weightedSupConLoss':
                division = 'val'
                loss = self.loss_fn(features, exon_names, division)
            else:
                loss = self.loss_fn(features)
        else:
            if len(z_views) != 2:
                raise ValueError(
                    f"{self.loss_fn.__class__.__name__} only supports 2 views per sample "
                    f"(got {len(z_views)} views). If you want more, use SupConLoss."
                )
            loss = self.loss_fn(z_views[0], z_views[1])

        self.log(
            'val_loss',
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=z_views[0].shape[0]
        )
        step_time = time.time() - start

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

