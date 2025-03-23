import torch
import torch.nn as nn
import lightning.pytorch as pl
from hydra.utils import instantiate
from torchmetrics import R2Score
import time

class PSIRegressionModel(pl.LightningModule):
    def __init__(self, encoder, config):
        super().__init__()
        # self.save_hyperparameters(ignore=['encoder'])

        self.encoder = encoder
        self.config = config

        if self.config.aux_models.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # if hasattr(encoder, "output_dim"):
        #     encoder_output_dim = encoder.output_dim
        # else:
        #     print("⚠️ Warning: `encoder.output_dim` not defined, inferring from dummy input.")
        #     dummy_input = torch.randint(0, 4, (1, self.config.dataset.seq_len))
        #     with torch.no_grad():
        #         dummy_output = encoder(dummy_input)
        #         encoder_output_dim = dummy_output.shape[-1]
        #     print(f"Inferred encoder output_dim = {encoder_output_dim}")

        self.regressor = nn.Linear(config.model.hidden_dim, config.aux_models.output_dim)

        # Instantiate loss and metrics via Hydra
        self.loss_fn = instantiate(config.loss)

        self.metric_fns = []
        for metric in config.task.metrics:
            if metric == "r2_score":
                self.metric_fns.append(R2Score())

    
    def forward(self, x):
        features = self.encoder(x)
        return self.regressor(features.mean(dim=1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x).squeeze()

        # inf masking
        if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
            print(f"❌ y_pred contains NaN or Inf at batch {batch_idx}")
            print("y_pred:", y_pred)

        if torch.isnan(y).any() or torch.isinf(y).any():
            print(f"❌ y contains NaN or Inf at batch {batch_idx}")
            print("y:", y)

        # Combine masks for any invalid predictions or targets
        valid_mask = ~(torch.isnan(y_pred) | torch.isinf(y_pred) | torch.isnan(y) | torch.isinf(y))
        # Filter out invalid values
        y_pred = y_pred[valid_mask]
        y = y[valid_mask]
        if y.numel() == 0:
            print(f"❌ Entire batch {batch_idx} is invalid — skipping.")

        
        loss = self.loss_fn(y_pred, y)

        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        for metric_fn in self.metric_fns:
            # value = metric_fn(y_pred, y)
            # print(f"🔍 Metric ({metric_fn.__class__.__name__}): {value.item()}")
            self.log(f"train_{metric_fn.__class__.__name__}", metric_fn(y_pred, y), on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x).squeeze()

        # inf masking
        if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
            print(f"❌ y_pred contains NaN or Inf at batch {batch_idx}")
            print("y_pred:", y_pred)

        if torch.isnan(y).any() or torch.isinf(y).any():
            print(f"❌ y contains NaN or Inf at batch {batch_idx}")
            print("y:", y)

        # Combine masks for any invalid predictions or targets
        valid_mask = ~(torch.isnan(y_pred) | torch.isinf(y_pred) | torch.isnan(y) | torch.isinf(y))
        # Filter out invalid values
        y_pred = y_pred[valid_mask]
        y = y[valid_mask]
        if y.numel() == 0:
            print(f"❌ Entire batch {batch_idx} is invalid — skipping.")

        loss = self.loss_fn(y_pred, y)

        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        for metric_fn in self.metric_fns:
            self.log(f"val_{metric_fn.__class__.__name__}", metric_fn(y_pred, y), on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        gpu_memory = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
        reserved_memory = torch.cuda.memory_reserved(0) / 1e9 if torch.cuda.is_available() else 0
        peak_memory = torch.cuda.max_memory_reserved(0) / 1e9 if torch.cuda.is_available() else 0

        self.log("epoch_time", epoch_time, prog_bar=True, sync_dist=True)
        self.log("gpu_memory_usage", gpu_memory, prog_bar=True, sync_dist=True)
        self.log("gpu_reserved_memory", reserved_memory, prog_bar=True, sync_dist=True)
        self.log("gpu_peak_memory", peak_memory, prog_bar=True, sync_dist=True)

        print(f"\nEpoch {self.current_epoch} took {epoch_time:.2f} seconds.")
        print(f"GPU Memory Used: {gpu_memory:.2f} GB, Reserved: {reserved_memory:.2f} GB, Peak: {peak_memory:.2f} GB")

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            print(f"Setting up training for {self.config.task._name_}")
        if stage == 'validate' or stage is None:
            print(f"Setting up validation for {self.config.task._name_}")

    def configure_optimizers(self):
        return instantiate(self.config.optimizer, params=self.parameters())
    



    
    
    # def forward(self, x):
    #     features = self.encoder(x)
    #     return self.regressor(features)

    # def step(self, batch, stage):
    #     x, y = batch
    #     y_pred = self(x).squeeze()
    #     loss = self.loss_fn(y_pred, y)
    #     self.log(f"{stage}_loss", loss, prog_bar=(stage == "val"))

    #     for metric_fn in self.metric_fns:
    #         metric_val = metric_fn(y_pred, y)
    #         self.log(f"{stage}_{metric_fn.__class__.__name__}", metric_val, prog_bar=(stage == "val"))

    #     return loss

    # def training_step(self, batch, batch_idx):
    #     return self.step(batch, "train")

    # def validation_step(self, batch, batch_idx):
    #     return self.step(batch, "val")

    # def configure_optimizers(self):
    #     return instantiate(self.config.optimizer, params=self.parameters())





# import torch
# import torch.nn as nn
# import lightning.pytorch as pl
# from hydra.utils import instantiate
# from src.model.simclr import get_simclr_model
# from torchmetrics.regression import R2Score

# class PSIRegressionModel(pl.LightningModule):
#     def __init__(self, encoder, config, freeze_encoder=True):
#         super().__init__()
#         self.encoder = encoder
#         self.regressor = nn.Linear(encoder.output_dim, 1)  # Predict a single PSI value
#         self.config = config
#         self.loss_fn = nn.MSELoss()  # Regression loss
#         self.r2_score = R2Score()

#         # If doing Linear Probing, freeze the encoder
#         if freeze_encoder:
#             for param in self.encoder.parameters():
#                 param.requires_grad = False

#     def forward(self, x):
#         features = self.encoder(x)
#         return self.regressor(features)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_pred = self.forward(x).squeeze()  # Ensure correct shape
#         loss = self.loss_fn(y_pred, y)
#         self.log("train_loss", loss)

#         r2 = self.r2_score(y_pred, y)
#         self.log("train_r2", r2)

#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_pred = self.forward(x).squeeze()
#         loss = self.loss_fn(y_pred, y)
#         r2 = self.r2_score(y_pred, y)

#         self.log("val_loss", loss, prog_bar=True)
#         self.log("val_r2", r2, prog_bar=True)

#         return loss

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=self.config.optimizer.lr)

