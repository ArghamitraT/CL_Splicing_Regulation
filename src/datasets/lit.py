import torch
import hydra
from transformers import AutoTokenizer
import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader
from src.datasets.base import NucleotideSequencePairDataset
from src.datasets.introns_alignment import IntronsDataset

class DummyDataModule(pl.LightningDataModule):
    def __init__(self, seq_len: int, num_pairs: int, batch_size: int, tokenizer_name: str):
        super().__init__()
        self.seq_len = seq_len
        self.num_pairs = num_pairs
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def generate_nucleotide_sequence(self, length):
        nucleotides = ['A', 'T', 'G', 'C']
        return ''.join(np.random.choice(nucleotides) for _ in range(length))

    def prepare_data(self):
        # Generate the sequences for training/validation/testing
        self.sequences_1 = np.array([self.generate_nucleotide_sequence(self.seq_len) for _ in range(self.num_pairs)])
        self.sequences_2 = np.array([self.generate_nucleotide_sequence(self.seq_len) for _ in range(self.num_pairs)])

    def setup(self, stage=None):
        # Splitting the sequences into train/val/test sets (e.g., 80% train, 10% val, 10% test)
        train_size = int(0.8 * self.num_pairs)
        val_size = int(0.1 * self.num_pairs)
        
        self.train_sequences_1 = self.sequences_1[:train_size]
        self.train_sequences_2 = self.sequences_2[:train_size]

        self.val_sequences_1 = self.sequences_1[train_size:train_size+val_size]
        self.val_sequences_2 = self.sequences_2[train_size:train_size+val_size]

        self.test_sequences_1 = self.sequences_1[train_size+val_size:]
        self.test_sequences_2 = self.sequences_2[train_size+val_size:]

    def train_dataloader(self):
        train_dataset = NucleotideSequencePairDataset(self.train_sequences_1, self.train_sequences_2, self.tokenizer)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        val_dataset = NucleotideSequencePairDataset(self.val_sequences_1, self.val_sequences_2, self.tokenizer)
        return DataLoader(val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        test_dataset = NucleotideSequencePairDataset(self.test_sequences_1, self.test_sequences_2, self.tokenizer)
        return DataLoader(test_dataset, batch_size=self.batch_size)
    
    
class IntronsDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.data_dir = config.dataset.data_dir
        self.batch_size = config.dataset.batch_size_per_device
        self.num_workers = config.dataset.num_workers
        self.train_ratio = config.dataset.train_ratio
        self.val_ratio = config.dataset.val_ratio
        self.test_ratio = config.dataset.test_ratio
        self.tokenizer = hydra.utils.instantiate(config.tokenizer)

    def prepare_data(self):
        # This is where we could download the data or pre-process it if needed.
        pass

    def setup(self, stage=None):
        # Split the dataset for train/val/test (e.g., 80% train, 10% val, 10% test)
        full_dataset = IntronsDataset(self.data_dir, self.tokenizer)
        dataset_size = len(full_dataset)
        train_size = int(self.train_ratio * dataset_size)
        val_size = int(self.val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size

        self.train_set, self.val_set, self.test_set = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=4, pin_memory=True)

