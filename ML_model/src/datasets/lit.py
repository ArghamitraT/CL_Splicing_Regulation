import torch
import hydra
from transformers import AutoTokenizer
import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, random_split
from src.datasets.base import NucleotideSequencePairDataset
from src.datasets.introns_alignment import ContrastiveIntronsDataset
import time

start = time.time()
def make_collate_fn(tokenizer, padding_strategy):
    def collate_fn(batch):
        from torch.utils.data import get_worker_info
        import os
        info = get_worker_info()
        # print(f"👷 Worker ID: {info.id if info else 'MAIN'}, PID: {os.getpid()}")

        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            raise ValueError("All items in batch were None")

        view1_sequences = [item[0] for item in batch]
        view2_sequences = [item[1] for item in batch]
        exon_ids = [item[2] for item in batch]
        
        token_start = time.time()
        if callable(tokenizer) and not hasattr(tokenizer, "vocab_size"):  # 
            view1 = tokenizer(view1_sequences)
            view2 = tokenizer(view2_sequences)
            output = view1, view2
        elif callable(tokenizer):  # HuggingFace-style
            view1 = tokenizer(view1_sequences, return_tensors='pt', padding=padding_strategy).input_ids
            view2 = tokenizer(view2_sequences, return_tensors='pt', padding=padding_strategy).input_ids
            output = view1, view2, exon_ids
        else:
            output = view1_sequences, view2_sequences
        # print(f"👷 Worker {info.id if info else 'MAIN'}: Collate time = {time.time() - start:.2f}s")
        # token_time = time.time() - token_start
        # total_time = time.time() - start
        # print(f"🧬 Tokenization took {token_time:.4f}s | 👷 Collate total time: {total_time:.4f}s")

        return output


    return collate_fn


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
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        val_dataset = NucleotideSequencePairDataset(self.val_sequences_1, self.val_sequences_2, self.tokenizer)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        test_dataset = NucleotideSequencePairDataset(self.test_sequences_1, self.test_sequences_2, self.tokenizer)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
    

class ContrastiveIntronsDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.train_file = config.dataset.train_data_file
        self.val_file = config.dataset.val_data_file
        self.test_file = config.dataset.test_data_file
        # self.exon_names_path = config.dataset.exon_names_path
        self.batch_size = config.dataset.batch_size_per_device
        self.num_workers = config.dataset.num_workers
        self.tokenizer = hydra.utils.instantiate(config.tokenizer)
        self.padding_strategy = config.tokenizer.padding
        self.collate_fn = make_collate_fn(self.tokenizer, self.padding_strategy)


        # self.data_file = config.dataset.data_file
        # self.exon_names_path = config.dataset.exon_names_path
        # self.batch_size = config.dataset.batch_size_per_device
        # self.num_workers = config.dataset.num_workers
        # self.train_ratio = config.dataset.train_ratio
        # self.val_ratio = config.dataset.val_ratio
        # self.test_ratio = config.dataset.test_ratio
        # self.tokenizer = hydra.utils.instantiate(config.tokenizer)
        # self.padding_strategy = config.tokenizer.padding
        # self.collate_fn = make_collate_fn(self.tokenizer, self.padding_strategy)


    def prepare_data(self):
        # Data preparation steps if needed, such as data checks or downloads.
        pass

    def setup(self, stage=None):
        # # Create the full dataset
        # full_dataset = ContrastiveIntronsDataset(
        #     data_file=self.data_file,
        #     exon_names_path=self.exon_names_path)
        
        # # Determine sizes for train, validation, and test sets
        # dataset_size = len(full_dataset)
        # train_size = int(self.train_ratio * dataset_size)
        # val_size = int(self.val_ratio * dataset_size)
        # test_size = dataset_size - train_size - val_size

        # # Split dataset into train, validation, and test
        # self.train_set, self.val_set, self.test_set = random_split(
        #     full_dataset,
        #     [train_size, val_size, test_size]
        # )  
        # Create dataset from pre-split pickle files
        # self.train_set = ContrastiveIntronsDataset(
        #     data_file=self.train_file,
        #     exon_names_path=self.exon_names_path
        # )

        # self.val_set = ContrastiveIntronsDataset(
        #     data_file=self.val_file,
        #     exon_names_path=self.exon_names_path
        # )

        # self.test_set = ContrastiveIntronsDataset(
        #     data_file=self.test_file,
        #     exon_names_path=self.exon_names_path
        # )
        self.train_set = ContrastiveIntronsDataset(
            data_file=self.train_file
        )

        self.val_set = ContrastiveIntronsDataset(
            data_file=self.val_file
            )

        self.test_set = ContrastiveIntronsDataset(
            data_file=self.test_file
             )
    
    # def collate_fn(self, batch):

    #     from torch.utils.data import get_worker_info
    #     info = get_worker_info()
    #     print(f"👷 Worker ID: {info.id if info else 'MAIN'}, PID: {os.getpid()}")


    #     # Remove samples that returned None (e.g., missing species)
    #     batch = [item for item in batch if item is not None]

    #     if len(batch) == 0:
    #         raise ValueError("All items in batch were None — likely due to missing species.")

    #     # Separate all augmentations from batch into two lists
    #     view1_sequences = [item[0] for item in batch]
    #     view2_sequences = [item[1] for item in batch]
        
    #     if callable(self.tokenizer):  # HuggingFace tokenizer or similar
    #         view1 = self.tokenizer(
    #             view1_sequences,
    #             return_tensors='pt',
    #             padding=self.padding_strategy,
    #         ).input_ids
    #         view2 = self.tokenizer(
    #             view2_sequences,
    #             return_tensors='pt',
    #             padding=self.padding_strategy,
    #         ).input_ids
    #         return view1, view2

    #     else:
    #         # No tokenizer: pass raw strings to encoder (interpretable encoder will preprocess)
    #         return view1_sequences, view2_sequences
            
        
    def train_dataloader(self):
        return DataLoader(
            self.train_set, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            collate_fn=self.collate_fn, 
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            collate_fn=self.collate_fn, 
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            collate_fn=self.collate_fn, 
            pin_memory=True
        )