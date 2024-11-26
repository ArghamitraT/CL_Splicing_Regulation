import pandas as pd
import torch
from torch.utils.data import Dataset
import random

class ConstitutiveIntronsDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (str): Path to the .csv file containing the dataset.
            seq_length (int): Optional fixed sequence length for truncation or padding.
        """
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing paired augmentations and the binary label.
        """
        # Select the current row
        row = self.data.iloc[idx]
        
        # Extract intronic sequence and label
        intronic_sequence = row["Intronic_Sequence"]
        label = row["Constitutive"]

        return intronic_sequence,label