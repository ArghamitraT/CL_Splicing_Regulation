import torch
import torch.nn as nn
import lightning.pytorch as pl
from hydra.utils import instantiate
from src.model.simclr import get_simclr_model
from torchmetrics.regression import R2Score

class PSIRegressionModel(pl.LightningModule):
    def __init__(self, encoder, config, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        self.regressor = nn.Linear(encoder.output_dim, 1)  # Predict a single PSI value
        self.config = config
        self.loss_fn = nn.MSELoss()  # Regression loss
        self.r2_score = R2Score()

        # If doing Linear Probing, freeze the encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.encoder(x)
        return self.regressor(features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x).squeeze()  # Ensure correct shape
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss)

        r2 = self.r2_score(y_pred, y)
        self.log("train_r2", r2)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x).squeeze()
        loss = self.loss_fn(y_pred, y)
        r2 = self.r2_score(y_pred, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_r2", r2, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.optimizer.lr)

