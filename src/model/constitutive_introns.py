import torch
from torch import nn
import os
import yaml
import lightning.pytorch as pl
from torchmetrics import Accuracy, Recall, Precision, AUROC

from src.model.lit import LitModel
from src.embedder.ntv2 import NTv2Embedder
from src.embedder.resnet import ResNet1D

embedder_mapping = {"nucleotide-transformer-v2": NTv2Embedder, "resnet": ResNet1D}

class ConstitutiveIntronsClassifier(pl.LightningModule):
    def __init__(self,
                 trained_simclr_ckpt=None,
                 encoder_name=None,
                 classification_head_embedding_dimension=512,
                 num_classes=2,
                 lr=1e-4,
                 use_checkpoint=True,
                 config_dir="configs/embedder"):
        """
        Args:
            trained_simclr_ckpt: Path to the SimCLR checkpoint.
            encoder_name: Name of the encoder.
            classification_head_embedding_dimension: Embedding dimension for classification head.
            num_classes (int): Number of output classes.
            lr (float): Learning rate for the optimizer.
            use_checkpoint (bool): Whether to load the model from a checkpoint or initialize a new one.
            embedder_config: Configuration for initializing a new model.
        """
        super().__init__()
        self.save_hyperparameters()
        self.use_checkpoint = use_checkpoint
        self.encoder_name = encoder_name
        self.config_dir = config_dir
        self.classification_head_embedding_dimension = classification_head_embedding_dimension
        self.lr = lr
        
        # Load embedder configuration from YAML if not provided
        config_path = os.path.join(config_dir, f"{encoder_name}.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found for encoder `{encoder_name}` at `{config_path}`.")
        with open(config_path, "r") as f:
            self.embedder_config = yaml.safe_load(f)

        self.criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
        self.num_classes = num_classes

        # Instantiate metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_recall = Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average='macro')

        self.train_precision = Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average='macro')

        self.train_auroc = AUROC(task="multiclass", num_classes=num_classes, average='macro')
        self.val_auroc = AUROC(task="multiclass", num_classes=num_classes, average='macro')
        self.test_auroc = AUROC(task="multiclass", num_classes=num_classes, average='macro')

        # Initialize the encoder and classification head
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

        self.classification_head = nn.Sequential(
            nn.Linear(self.last_embedding_dimension, self.classification_head_embedding_dimension),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.classification_head_embedding_dimension, self.num_classes)
        )
        
            
    def setup(self, stage=None):
        """
        Setup function for model training. This function is called at the beginning of training
        and validation, and it allows the model to prepare its environment for the given stage.

        Args:
            stage (str): Either 'fit' for training or 'validate' for validation.
        """
        if stage == 'fit':
            self.encoder.train()
            self.classification_head.train()
        if stage == 'test' or stage is None:
            self.encoder.eval()
            self.classification_head.eval()
        

    def forward(self, input_ids):
        """
        Forward pass through the encoder backbone and classification head.
        """
        features = self.encoder(input_ids)
        embedding = features.mean(dim=1)
        logits = self.classification_head(embedding)
        return logits

    def training_step(self, batch, batch_idx):
        """
        Training step.
        """
        input_ids, labels = batch
        logits = self(input_ids)
        loss = self.criterion(logits.squeeze(), labels)
        # For classification metrics that require discrete predictions
        preds = torch.argmax(logits, dim=1) if self.num_classes > 1 else torch.sigmoid(logits) > 0.5
        
        # Update metrics
        self.train_accuracy.update(preds, labels)
        self.train_recall.update(preds, labels)
        self.train_precision.update(preds, labels)
        self.train_auroc.update(logits, labels)  # Use raw logits for AUROC
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        """
        input_ids, labels = batch
        logits = self(input_ids)
        loss = self.criterion(logits.squeeze(), labels)
        
         # For classification metrics that require discrete predictions
        preds = torch.argmax(logits, dim=1) if self.num_classes > 1 else torch.sigmoid(logits) > 0.5

        # Update metrics
        self.val_accuracy.update(preds, labels)
        self.val_recall.update(preds, labels)
        self.val_precision.update(preds, labels)
        self.val_auroc.update(logits, labels)  # Use raw logits for AUROC
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step.
        """
        input_ids, labels = batch
        logits = self(input_ids)
        loss = self.criterion(logits.squeeze(), labels)
        # For classification metrics that require discrete predictions
        preds = torch.argmax(logits, dim=1) if self.num_classes > 1 else torch.sigmoid(logits) > 0.5

        # Update metrics
        self.test_accuracy.update(preds, labels)
        self.test_recall.update(preds, labels)
        self.test_precision.update(preds, labels)
        self.test_auroc.update(logits, labels)  # Use raw logits for AUROC

        self.log("test_loss", loss)
        return loss

    def on_train_epoch_end(self):
        self.log("train_accuracy", self.train_accuracy.compute(), on_epoch=True, prog_bar=True)
        self.log("train_recall", self.train_recall.compute(), on_epoch=True, prog_bar=True)
        self.log("train_precision", self.train_precision.compute(), on_epoch=True, prog_bar=True)
        self.log("train_auroc", self.train_auroc.compute(), on_epoch=True, prog_bar=True)

        self.train_accuracy.reset()
        self.train_recall.reset()
        self.train_precision.reset()
        self.train_auroc.reset()

    def on_validation_epoch_end(self):
        self.log("val_accuracy", self.val_accuracy.compute(), on_epoch=True, prog_bar=True)
        self.log("val_recall", self.val_recall.compute(), on_epoch=True, prog_bar=True)
        self.log("val_precision", self.val_precision.compute(), on_epoch=True, prog_bar=True)
        self.log("val_auroc", self.val_auroc.compute(), on_epoch=True, prog_bar=True)

        self.val_accuracy.reset()
        self.val_recall.reset()
        self.val_precision.reset()
        self.val_auroc.reset()

    def on_test_epoch_end(self):
        self.log("test_accuracy", self.test_accuracy.compute(), on_epoch=True, prog_bar=True)
        self.log("test_recall", self.test_recall.compute(), on_epoch=True, prog_bar=True)
        self.log("test_precision", self.test_precision.compute(), on_epoch=True, prog_bar=True)
        self.log("test_auroc", self.test_auroc.compute(), on_epoch=True, prog_bar=True)

        self.test_accuracy.reset()
        self.test_recall.reset()
        self.test_precision.reset()
        self.test_auroc.reset()

    def configure_optimizers(self):
        """
        Configure the optimizer.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
