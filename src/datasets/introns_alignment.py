import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

class IntronsDataset(Dataset):
    def __init__(self, data_dir, tokenizer):
        self.data_dir = data_dir
        self.tokenizer = tokenizer  # Pass a tokenizer to the dataset
        self.files = sorted([
            f for f in os.listdir(data_dir)
            if f.startswith('intron_') and f.endswith('_matrix.npy')
        ])

    def __len__(self):
        return len(self.files)

    def encode_sequence(self, sequence):
        # Tokenize the sequence using the tokenizer passed to the dataset
        return self.tokenizer(sequence, return_tensors='pt', padding='max_length', truncation=True).input_ids

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.data_dir, file_name)
        gene_data = np.load(file_path)  # Shape: (sequence_length, 100)
        
        # Split into two views and tokenize using the tokenizer
        view1 = torch.cat([self.encode_sequence(''.join(seq)) for seq in gene_data[:50, :]], dim=0)  # First 50 species
        view2 = torch.cat([self.encode_sequence(''.join(seq)) for seq in gene_data[50:, :]], dim=0)  # Last 50 species
        
        return view1, view2