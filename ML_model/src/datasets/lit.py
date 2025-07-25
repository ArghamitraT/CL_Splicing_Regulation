import torch
import hydra
from transformers import AutoTokenizer
import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, random_split
# from src.datasets.base import NucleotideSequencePairDataset
from src.datasets.introns_alignment import ContrastiveIntronsDataset
import time

start = time.time()
# def make_collate_fn(tokenizer, padding_strategy):
#     def collate_fn(batch):
#         from torch.utils.data import get_worker_info
#         import os
#         info = get_worker_info()
#         # print(f"👷 Worker ID: {info.id if info else 'MAIN'}, PID: {os.getpid()}")

#         batch = [item for item in batch if item is not None]
#         if len(batch) == 0:
#             raise ValueError("All items in batch were None")

#         view1_sequences = [item[0] for item in batch]
#         view2_sequences = [item[1] for item in batch]
#         exon_ids = [item[2] for item in batch]
        
#         token_start = time.time()
#         if callable(tokenizer) and not hasattr(tokenizer, "vocab_size"):  # 
#             view1 = tokenizer(view1_sequences)
#             view2 = tokenizer(view2_sequences)
#             output = view1, view2
#         elif callable(tokenizer):  # HuggingFace-style
#             view1 = tokenizer(view1_sequences, return_tensors='pt', padding=padding_strategy).input_ids
#             view2 = tokenizer(view2_sequences, return_tensors='pt', padding=padding_strategy).input_ids
#             output = view1, view2, exon_ids
#         else:
#             output = view1_sequences, view2_sequences
#         # print(f"👷 Worker {info.id if info else 'MAIN'}: Collate time = {time.time() - start:.2f}s")
#         # token_time = time.time() - token_start
#         # total_time = time.time() - start
#         # print(f"🧬 Tokenization took {token_time:.4f}s | 👷 Collate total time: {total_time:.4f}s")

#         return output


#     return collate_fn



def make_collate_fn(tokenizer, padding_strategy):
    def collate_fn(batch):
        from torch.utils.data import get_worker_info
        import os
        info = get_worker_info()
        # print(f"👷 Worker ID: {info.id if info else 'MAIN'}, PID: {os.getpid()}")

        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            raise ValueError("All items in batch were None")

        exon_ids = [item[1] for item in batch]
        min_n_views = min(len(item[0]) for item in batch)
        # For each sample, take only the first min_n_views augmentations
        exon_ids = [item[1] for item in batch]
        view_lists = [[item[0][i] for item in batch] for i in range(min_n_views)]
        token_start = time.time()
        if callable(tokenizer) and not hasattr(tokenizer, "vocab_size"):  # 
            tokenized_views = [tokenizer(view) for view in view_lists]
            output = (*tokenized_views, exon_ids)
        elif callable(tokenizer):  # HuggingFace-style
            tokenized_views = [
                tokenizer(view, return_tensors='pt', padding=padding_strategy).input_ids
                for view in view_lists
            ]
            output = (*tokenized_views, exon_ids)
        else:
            output = (*view_lists, exon_ids)
        
        # print(f"👷 Worker {info.id if info else 'MAIN'}: Collate time = {time.time() - start:.2f}s")
        # token_time = time.time() - token_start
        # total_time = time.time() - start
        # print(f"🧬 Tokenization took {token_time:.4f}s | 👷 Collate total time: {total_time:.4f}s")

        return output


    return collate_fn


class ContrastiveIntronsDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.train_file = config.dataset.train_data_file
        self.val_file = config.dataset.val_data_file
        self.test_file = config.dataset.test_data_file
        self.n_augmentations = config.dataset.n_augmentations #(AT)
        self.batch_size = config.dataset.batch_size_per_device
        self.num_workers = config.dataset.num_workers
        self.tokenizer = hydra.utils.instantiate(config.tokenizer)
        self.padding_strategy = config.tokenizer.padding
        self.collate_fn = make_collate_fn(self.tokenizer, self.padding_strategy)

    def prepare_data(self):
        # Data preparation steps if needed, such as data checks or downloads.
        pass

    def setup(self, stage=None):
        
        self.train_set = ContrastiveIntronsDataset(
            data_file=self.train_file,
            n_augmentations=self.n_augmentations #(AT)
        )

        self.val_set = ContrastiveIntronsDataset(
            data_file=self.val_file,
            n_augmentations=self.n_augmentations #(AT)
            )

        self.test_set = ContrastiveIntronsDataset(
            data_file=self.test_file,
            n_augmentations=self.n_augmentations #(AT)
             )
    
        
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