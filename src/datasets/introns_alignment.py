import os
import numpy as np
import torch
import random
import pickle
from torch.utils.data import Dataset
    
class ContrastiveIntronsDataset(Dataset):
    def __init__(self, data_file, exon_names_path):
        # Load the merged data and exon names
        with open(data_file, 'rb') as file:
            self.data = pickle.load(file)
        with open(exon_names_path, 'r') as file:
            self.exon_names = [line.strip() for line in file]
    
    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        # Randomly select an exon
        exon_name = self.exon_names[idx]
        intronic_sequences = self.data[exon_name]
                
        # Randomly sample two augmentations (intronic sequences)
        species_sample = random.sample(intronic_sequences.keys(), 2)
        # Retrieve the sequences for the sampled species
        augmentation1 = intronic_sequences[species_sample[0]]
        augmentation2 = intronic_sequences[species_sample[1]]
        
        return augmentation1, augmentation2