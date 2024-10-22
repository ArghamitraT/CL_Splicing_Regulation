from src.datasets.base import NucleotideSequencePairDataset
from transformers import AutoTokenizer
import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader

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
