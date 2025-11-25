import os
import numpy as np
import torch
import random
import pickle
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from .utility import (
    get_windows_with_padding
)

class ContrastiveIntronsDataset(Dataset):
    def __init__(self, data_file, n_augmentations=2, embedder=None, fixed_species=0, len_5p=300, len_3p=300):
        
        with open(data_file, 'rb') as file:
            self.data = pickle.load(file)
        self.exon_names = list(self.data.keys())  
        self.exon_name_to_id = {name: i for i, name in enumerate(self.exon_names)}
        self.n_augmentations = n_augmentations   # <-- store as an attribute
        self.embedder = embedder
        
        self.len_5p = len_5p
        self.len_exon = 100
        self.len_3p = len_3p
        self.tissue_acceptor_intron = 300
        self.tissue_donor_intron = 300
        
        self.tissue_acceptor_exon = 100
        self.tissue_donor_exon = 100

        
    
    def __len__(self):

        return len(self.data)
        

    def __getitem__(self, idx):
    
        exon_name = self.exon_names[idx]
        exon_id = self.exon_name_to_id[exon_name]
        intronic_sequences = self.data[exon_name]
        exon_id = self.exon_name_to_id[exon_name]
        # original fallback behavior
        all_species = list(intronic_sequences.keys())
        n_available = len(all_species)
        species_sample = all_species
        augmentations = []

        for sp in species_sample:
            intron_5p = intronic_sequences[sp]['5p']
            exon_seq = intronic_sequences[sp]['exon']
            intron_3p = intronic_sequences[sp]['3p']
            full_seq = intron_5p + exon_seq + intron_3p
            windows = get_windows_with_padding(self.tissue_acceptor_intron, self.tissue_donor_intron, self.tissue_acceptor_exon, self.tissue_donor_exon, full_seq, overhang = (self.len_3p, self.len_5p))
            augmentations.append({'acceptor': windows['acceptor'], 'donor': windows['donor']})

        return augmentations, exon_id, exon_name  # return exon_name for debugging



