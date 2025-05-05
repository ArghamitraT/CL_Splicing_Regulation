import torch
import torch.nn as nn
import lightning.pytorch as pl
from hydra.utils import instantiate
from torchmetrics import R2Score
import time
from scipy.stats import spearmanr

class PSIRegressionModel(pl.LightningModule):
    def __init__(self, encoder, config):
        super().__init__()
        # self.save_hyperparameters(ignore=['encoder'])

        self.encoder = encoder
        self.config = config

        if self.config.aux_models.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if hasattr(encoder, "output_dim"):
            encoder_output_dim = encoder.output_dim
        else:
            print("⚠️ Warning: `encoder.output_dim` not defined, inferring from dummy input.")
            dummy_input = torch.randint(0, 4, (1, self.config.dataset.seq_len))
            with torch.no_grad():
                dummy_output = encoder(dummy_input)
                encoder_output_dim = dummy_output.shape[-1]
            print(f"Inferred encoder output_dim = {encoder_output_dim}")

        
        # self.regressor = nn.Sequential(nn.Linear(201*encoder_output_dim, config.aux_models.hidden_dim),
        #                                nn.ReLU(),
        #                                nn.Linear(config.aux_models.hidden_dim, config.aux_models.output_dim))

        # self.regressor = nn.Linear(201, config.aux_models.output_dim)
        # self.regressor = nn.Sequential(nn.Linear(201, config.aux_models.hidden_dim),
        #                                nn.ReLU(),
        #                                nn.Linear(config.aux_models.hidden_dim, config.aux_models.output_dim))

        self.regressor = nn.Sequential(nn.Linear(encoder_output_dim, config.aux_models.hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(config.aux_models.hidden_dim, config.aux_models.output_dim))



        # Instantiate loss and metrics via Hydra
        self.loss_fn = instantiate(config.loss)

        self.metric_fns = []
        for metric in config.task.metrics:
            if metric == "r2_score":
                self.metric_fns.append(R2Score())

    
    def forward(self, x):
        features = self.encoder(x)
        # features = features.flatten(start_dim=1)
        # return self.regressor(features)
        # return self.regressor(features.mean(dim=2))
        # return self.regressor(x.float())
        return self.regressor(features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x).squeeze()
        # y_pred = self(x)

        # # Combine masks for any invalid predictions or targets
        # valid_mask = ~(torch.isnan(y_pred) | torch.isinf(y_pred) | torch.isnan(y) | torch.isinf(y))
        # # Filter out invalid values
        # y_pred = y_pred[valid_mask]
        # y = y[valid_mask]
        loss = self.loss_fn(y_pred, y)

        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        for metric_fn in self.metric_fns:
            # value = metric_fn(y_pred, y)
            # print(f"🔍 Metric ({metric_fn.__class__.__name__}): {value.item()}")
            self.log(f"train_{metric_fn.__class__.__name__}", metric_fn(y_pred, y), on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # y_pred = self(x)
        y_pred = self(x).squeeze()

        # Combine masks for any invalid predictions or targets
        # valid_mask = ~(torch.isnan(y_pred) | torch.isinf(y_pred) | torch.isnan(y) | torch.isinf(y))
        # # Filter out invalid values
        # y_pred = y_pred[valid_mask]
        # y = y[valid_mask]
        loss = self.loss_fn(y_pred, y)

        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        for metric_fn in self.metric_fns:
            self.log(f"val_{metric_fn.__class__.__name__}", metric_fn(y_pred, y), on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x).squeeze()
        loss = self.loss_fn(y_pred, y)

        for metric_fn in self.metric_fns:
            self.log(f"test_{metric_fn.__class__.__name__}", metric_fn(y_pred, y), on_epoch=True, prog_bar=True, sync_dist=True)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        # Store for Spearman
        self.test_preds.append(y_pred.detach().cpu())
        self.test_targets.append(y.detach().cpu())

        return loss

    def on_test_epoch_start(self):
        self.test_preds = []
        self.test_targets = []

    def on_test_epoch_end(self):
        y_pred_all = torch.cat(self.test_preds).numpy()
        y_true_all = torch.cat(self.test_targets).numpy()

        rho, _ = spearmanr(y_true_all, y_pred_all)
        self.log("test_spearman", rho, prog_bar=True, sync_dist=True)
        print(f"\n🔬 Spearman ρ (test set): {rho:.4f}")


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
    



    