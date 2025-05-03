import os
import numpy as np
import torch
import random
import pickle
from torch.utils.data import Dataset

import inspect
import os

_warned_debug = False  # module-level flag

def debug_warning():
    global _warned_debug
    if not _warned_debug:
        frame = inspect.currentframe().f_back
        filename = os.path.basename(frame.f_code.co_filename)
        lineno = frame.f_lineno
        print(f"\033[1;31m⚠️ DEBUG MODE ENABLED in {filename}:{lineno} — Using fixed species views! REMEMBER TO REVERT!\033[0m")
        _warned_debug = True

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

    # def __getitem__(self, idx):

    #     debug_warning()  # will only print the first time
    #     # Randomly select an exon
    #     exon_name = self.exon_names[idx]
    #     intronic_sequences = self.data[exon_name]
                
    #     # Randomly sample two augmentations (intronic sequences)
    #     species_1, species_2 = 'hg38', 'panTro4'
    #     if species_1 not in intronic_sequences or species_2 not in intronic_sequences:
    #         return None  # Let collate_fn handle skipping this

    #     augmentation1 = intronic_sequences[species_1]
    #     augmentation2 = intronic_sequences[species_2]

    #     return augmentation1, augmentation2
