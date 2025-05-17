import torch
import hydra
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
import pickle
import lightning.pytorch as pl

class PSIRegressionDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=201):
        """
        Dataset for PSI Regression.

        Args:
            data_file (str): Path to the pickle file containing PSI values and sequences.
            tokenizer_name (str): Name of the tokenizer to use.
            max_length (int): Max sequence length for padding.
        """
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer = tokenizer

        # Load data from pickle file
        with open(data_file, "rb") as f:
            self.data = pickle.load(f)

        self.max_length = max_length
        self.entries = list(self.data.items())  # Convert dictionary to list format

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry_id, entry = self.entries[idx]
        psi_value = entry["psi_val"]
        sequence = entry["hg38"]

        if callable(self.tokenizer) and not hasattr(self.tokenizer, "vocab_size"):
            # Custom one-hot tokenizer
            encoded_seq = self.tokenizer([sequence])[0]  # shape: (C, L)
        else:
            # HuggingFace-style tokenizer
            encoded_seq = self.tokenizer(
                sequence,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            ).input_ids.squeeze(0)  # shape: (L,)

        return encoded_seq, torch.tensor(psi_value, dtype=torch.float32), entry_id

class PSIRegressionDataModule(pl.LightningDataModule):
    def __init__(self, config):
        """
        PyTorch Lightning DataModule for PSI Regression.

        Args:
            config (OmegaConf): Config object with dataset parameters.
        """
        super().__init__()
        self.train_file = config.dataset.train_file
        self.val_file = config.dataset.val_file
        self.test_file = config.dataset.test_file
        self.batch_size = config.dataset.batch_size_per_device
        self.num_workers = config.dataset.num_workers
        self.tokenizer = hydra.utils.instantiate(config.tokenizer)
        
    def setup(self, stage=None):
        """
        Load dataset and split into train/val/test sets.
        """
        self.train_set = PSIRegressionDataset(self.train_file, self.tokenizer)
        self.val_set = PSIRegressionDataset(self.val_file, self.tokenizer)
        self.test_set = PSIRegressionDataset(self.test_file, self.tokenizer)
    
    
    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )
