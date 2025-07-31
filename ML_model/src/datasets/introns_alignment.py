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
    def __init__(self, data_file, n_augmentations=2, embedder=None):
        # Load the merged data and exon names
        with open(data_file, 'rb') as file:
            self.data = pickle.load(file)
        self.exon_names = list(self.data.keys())  
        self.exon_name_to_id = {name: i for i, name in enumerate(self.exon_names)}
        self.n_augmentations = n_augmentations   # <-- store as an attribute
        self.embedder = embedder

        # Fixed lengths for MTSplice windowing
        self.len_5p = 200
        self.len_exon = 100
        self.len_3p = 200
        self.tissue_acceptor_intron = 300
        self.tissue_acceptor_exon = 100
        self.tissue_donor_intron = 300
        self.tissue_donor_exon = 100

        # Fixed seed
        # debug_warning() 
        # self.seed = 42
        #  # will only print the first time
        # self.random_state = random.Random(self.seed)

        
    
    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):

        def get_windows_with_padding(full_seq, len_5p, len_exon, len_3p,
                             tissue_acceptor_intron, tissue_acceptor_exon,
                             tissue_donor_exon, tissue_donor_intron):
            # Acceptors: region around exon start (3' splice site)
            acceptor_intron = len_3p  # region before the exon (3' intron)
            # Donor: region around exon end (5' splice site)
            donor_intron = len_5p     # region after the exon (5' intron)

            # Get acceptor window
            acceptor_start = len_5p + 0 - tissue_acceptor_intron
            acceptor_end = len_5p + tissue_acceptor_exon

            # Pad acceptor if needed
            seq_acceptor = full_seq[max(0, acceptor_start):acceptor_end]
            if acceptor_start < 0:
                seq_acceptor = "N" * abs(acceptor_start) + seq_acceptor
            if len(seq_acceptor) < tissue_acceptor_intron + tissue_acceptor_exon:
                seq_acceptor = seq_acceptor + "N" * (tissue_acceptor_intron + tissue_acceptor_exon - len(seq_acceptor))

            # Get donor window
            donor_end = len_5p + len_exon + tissue_donor_intron
            donor_start = len_5p + len_exon - tissue_donor_exon

            seq_donor = full_seq[donor_start:donor_end]
            if donor_end > len(full_seq):
                seq_donor = seq_donor + "N" * (donor_end - len(full_seq))
            if donor_start < 0:
                seq_donor = "N" * abs(donor_start) + seq_donor
            if len(seq_donor) < tissue_donor_exon + tissue_donor_intron:
                seq_donor = seq_donor + "N" * (tissue_donor_exon + tissue_donor_intron - len(seq_donor))

            return {
                'acceptor': seq_acceptor,
                'donor': seq_donor
            }
    
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
            # debug_warning() 
            # species_sample = self.random_state.sample(all_species, self.n_augmentations)

        # # Retrieve the sequences for the sampled species
        # augmentations = [intronic_sequences[sp] for sp in species_sample]
        augmentations = []

        for sp in species_sample:
            full_seq = intronic_sequences[sp]

            if self.embedder.name_or_path == "MTSplice":
        
                windows = get_windows_with_padding(
                    full_seq,
                    self.len_5p, self.len_exon, self.len_3p,
                    self.tissue_acceptor_intron, self.tissue_acceptor_exon,
                    self.tissue_donor_exon, self.tissue_donor_intron
                )
                # augmentations.append((windows['acceptor'], windows['donor']))
                augmentations.append({'acceptor': windows['acceptor'], 'donor': windows['donor']})

            else:
                augmentations.append(full_seq)

        return augmentations, exon_id


