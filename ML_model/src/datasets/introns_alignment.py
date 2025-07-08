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
        # print(f"\033[1;31m⚠️ DEBUG MODE ENABLED in {filename}:{lineno} — Using fixed species views! REMEMBER TO REVERT!\033[0m")
        print(f"\033[1;31m⚠️ DEBUG MODE ENABLED in {filename}:{lineno} — seed fixed! REMEMBER TO REVERT!\033[0m")
        _warned_debug = True

class ContrastiveIntronsDataset(Dataset):
    def __init__(self, data_file, n_augmentations=2):
        # Load the merged data and exon names
        with open(data_file, 'rb') as file:
            self.data = pickle.load(file)
        self.exon_names = list(self.data.keys())  
        self.exon_name_to_id = {name: i for i, name in enumerate(self.exon_names)}
        self.n_augmentations = n_augmentations   # <-- store as an attribute

    
    def __len__(self):
        return len(self.data)

    # (AT) old function where we randomly chose 2 augmentations
    
    # def __getitem__(self, idx):
    #     # Randomly select an exon
    #     exon_name = self.exon_names[idx]
    #     exon_id = self.exon_name_to_id[exon_name]

    #     intronic_sequences = self.data[exon_name]
                
    #     # Randomly sample two augmentations (intronic sequences)
    #     try:
    #         # debug_warning()
    #         # random.seed(42) 
    #         species_sample = random.sample(intronic_sequences.keys(), 2)
    #     except:
    #         print("only 1 species", exon_name)
    #     # Retrieve the sequences for the sampled species
    #     augmentation1 = intronic_sequences[species_sample[0]]
    #     augmentation2 = intronic_sequences[species_sample[1]]
        
    #     return augmentation1, augmentation2, exon_id


    def __getitem__(self, idx):
        # Get the exon name and id
        exon_name = self.exon_names[idx]
        exon_id = self.exon_name_to_id[exon_name]
        intronic_sequences = self.data[exon_name]

        # Number of available augmentations
        all_species = list(intronic_sequences.keys())
        n_available = len(all_species)

        # Decide how many augmentations to sample
        if self.n_augmentations == "all" or self.n_augmentations > n_available:
            species_sample = all_species  # take all available
        else:
            species_sample = random.sample(all_species, self.n_augmentations)

        # Retrieve the sequences for the sampled species
        augmentations = [intronic_sequences[sp] for sp in species_sample]

        return augmentations, exon_id






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
