import torch
from torch import nn
import os
import yaml
import lightning.pytorch as pl
from torchmetrics import R2Score
from src.model.lit import LitModel
from src.embedder.ntv2 import NTv2Embedder
from src.embedder.resnet import ResNet1D

# Embedder mapping for dynamic initialization
embedder_mapping = {"nucleotide-transformer-v2": NTv2Embedder, "resnet": ResNet1D}

class PsiLungIntronsRegressor(pl.LightningModule):
    def __init__(self,
                 trained_simclr_ckpt=None,
                 encoder_name=None,
                 regression_head_embedding_dimension=512,
                 lr=1e-4,
                 use_checkpoint=True,
                 config_dir="configs/embedder"):
        """
        Args:
            trained_simclr_ckpt: Path to the SimCLR checkpoint.
            encoder_name: Name of the encoder.
            regression_head_embedding_dimension: Embedding dimension for the regression head.
            lr (float): Learning rate for the optimizer.
            use_checkpoint (bool): Whether to load the model from a checkpoint or initialize a new one.
            config_dir: Directory where embedder configuration YAML files are stored.
        """
        super().__init__()
        self.save_hyperparameters()
        self.use_checkpoint = use_checkpoint
        self.encoder_name = encoder_name
        self.config_dir = config_dir
        self.regression_head_embedding_dimension = regression_head_embedding_dimension
        self.lr = lr

        # Load embedder configuration from YAML
        config_path = os.path.join(config_dir, f"{encoder_name}.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found for encoder `{encoder_name}` at `{config_path}`.")
        with open(config_path, "r") as f:
            self.embedder_config = yaml.safe_load(f)

        self.criterion = nn.MSELoss()

        # Instantiate R2 metrics
        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()

        # Initialize the encoder and regression head
        if self.use_checkpoint:
            if not trained_simclr_ckpt:
                raise ValueError("`trained_simclr_ckpt` must be provided when `use_checkpoint` is True.")
            self.trained_simclr = LitModel.load_from_checkpoint(trained_simclr_ckpt, strict=False)
            self.last_embedding_dimension = self.trained_simclr.model.encoder.get_last_embedding_dimension()
            self.encoder = self.trained_simclr.model.encoder
            del self.trained_simclr
        else:
            if encoder_name not in embedder_mapping:
                raise ValueError(f"Unknown encoder name `{encoder_name}`. Available options: {list(embedder_mapping.keys())}.")
            embedder_class = embedder_mapping[encoder_name]
            self.encoder = embedder_class(**self.embedder_config)
            self.last_embedding_dimension = self.encoder.get_last_embedding_dimension()

        self.regression_head = nn.Sequential(
            nn.Linear(self.last_embedding_dimension, self.regression_head_embedding_dimension),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.regression_head_embedding_dimension, 1)  # Single output for regression
        )

    def setup(self, stage=None):
        """
        Setup function for model training.
        """
        if stage == 'fit':
            self.encoder.train()
            self.regression_head.train()
        if stage == 'test' or stage is None:
            self.encoder.eval()
            self.regression_head.eval()

    def forward(self, input_ids):
        """
        Forward pass through the encoder backbone and regression head.
        """
        features = self.encoder(input_ids)
        embedding = features.mean(dim=1)
        output = self.regression_head(embedding)
        return output

    def training_step(self, batch, batch_idx):
        """
        Training step.
        """
        input_ids, labels = batch
        preds = self(input_ids).squeeze()
        loss = self.criterion(preds, labels)

        # Update and compute R2 for training
        self.train_r2.update(preds, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        """
        input_ids, labels = batch
        preds = self(input_ids).squeeze()
        loss = self.criterion(preds, labels)

        # Update R2 for validation
        self.val_r2.update(preds, labels)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step.
        """
        input_ids, labels = batch
        preds = self(input_ids).squeeze()
        loss = self.criterion(preds, labels)

        # Update R2 for testing
        self.test_r2.update(preds, labels)

        self.log("test_loss", loss)
        return loss

    def on_train_epoch_end(self):
        """
        Compute and log R2 at the end of training epoch.
        """
        train_r2 = self.train_r2.compute()
        self.log("train_r2", train_r2, on_epoch=True, prog_bar=True, sync_dist=True)
        self.train_r2.reset()

    def on_validation_epoch_end(self):
        """
        Compute and log R2 at the end of validation epoch.
        """
        val_r2 = self.val_r2.compute()
        self.log("val_r2", val_r2, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_r2.reset()

    def on_test_epoch_end(self):
        """
        Compute and log R2 at the end of test epoch.
        """
        test_r2 = self.test_r2.compute()
        self.log("test_r2", test_r2, on_epoch=True, prog_bar=True, sync_dist=True)
        self.test_r2.reset()

    def configure_optimizers(self):
        """
        Configure the optimizer.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
