import torch
import hydra
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
import pickle
import lightning.pytorch as pl

class PSIRegressionDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=201, mode="5p"):
        """
        Dataset for PSI Regression.

        Args:
            data_file (str): Path to the pickle file containing PSI values and sequences.
            tokenizer_name (str): Name of the tokenizer to use.
            max_length (int): Max sequence length for padding.
        """
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.tokenizer = tokenizer

        # Load data from pickle file
        with open(data_file, "rb") as f:
            self.data = pickle.load(f)

        self.max_length = max_length
        self.mode = mode
        self.entries = list(self.data.items())  # Convert dictionary to list format

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):

        exon_id, entry = self.entries[idx]
        psi_value = entry["psi_val"]

        if self.mode == "intronexon":
            seq_3p = entry["3p"]
            seq_5p = entry["5p"]
            seq_exon = self._process_exon(entry["exon"])

            return (
                self._tokenize(seq_5p),
                self._tokenize(seq_3p),
                self._tokenize(seq_exon),
            ), torch.tensor(psi_value, dtype=torch.float32), exon_id

        else:
            sequence = entry["hg38"]
            return self._tokenize(sequence), torch.tensor(psi_value, dtype=torch.float32), exon_id

    def _process_exon(self, exon_dict):
        start = exon_dict.get("start", "")
        end = exon_dict.get("end", "")
        
        # Pad start to 100 bp (right pad)
        start_padded = start.ljust(100, "N")
        
        # Pad end to 100 bp (left pad)
        end_padded = end.rjust(100, "N")
        
        # Concatenate start + end
        return start_padded + end_padded

    def _tokenize(self, seq):
        if callable(self.tokenizer) and not hasattr(self.tokenizer, "vocab_size"):
            return self.tokenizer([seq])[0]
        else:
            return self.tokenizer(
                seq,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            ).input_ids.squeeze(0)

        # entry_id, entry = self.entries[idx]
        # psi_value = entry["psi_val"]
        # sequence = entry["hg38"]

        # if callable(self.tokenizer) and not hasattr(self.tokenizer, "vocab_size"):
        #     # Custom one-hot tokenizer
        #     encoded_seq = self.tokenizer([sequence])[0]  # shape: (C, L)
        # else:
        #     # HuggingFace-style tokenizer
        #     encoded_seq = self.tokenizer(
        #         sequence,
        #         return_tensors="pt",
        #         padding="max_length",
        #         truncation=True,
        #         max_length=self.max_length,
        #     ).input_ids.squeeze(0)  # shape: (L,)

        # return encoded_seq, torch.tensor(psi_value, dtype=torch.float32), entry_id

class PSIRegressionDataModule(pl.LightningDataModule):
    def __init__(self, config):
        """
        PyTorch Lightning DataModule for PSI Regression.

        Args:
            config (OmegaConf): Config object with dataset parameters.
        """
        super().__init__()
        self.config = config
        self.mode = config.aux_models.mode  # "3p", "5p", or "intronexon"

        self.batch_size = config.dataset.batch_size_per_device
        self.num_workers = config.dataset.num_workers
        self.tokenizer = hydra.utils.instantiate(config.tokenizer)

        self.train_files = config.dataset.train_files
        self.val_files = config.dataset.val_files
        self.test_files = config.dataset.test_files

    def setup(self, stage=None):
        if self.mode == "3p":
            self.train_set = PSIRegressionDataset(self.train_files["3p"], self.tokenizer, mode=self.mode)
            self.val_set = PSIRegressionDataset(self.val_files["3p"], self.tokenizer, mode=self.mode)
            self.test_set = PSIRegressionDataset(self.test_files["3p"], self.tokenizer, mode=self.mode)

        elif self.mode == "5p":
            self.train_set = PSIRegressionDataset(self.train_files["5p"], self.tokenizer, mode=self.mode)
            self.val_set = PSIRegressionDataset(self.val_files["5p"], self.tokenizer, mode=self.mode)
            self.test_set = PSIRegressionDataset(self.test_files["5p"], self.tokenizer, mode=self.mode)

        elif self.mode == "intronexon":
            self.train_set = PSIRegressionDataset(self.train_files["intronexon"], self.tokenizer, mode=self.mode)
            self.val_set = PSIRegressionDataset(self.val_files["intronexon"], self.tokenizer, mode=self.mode)
            self.test_set = PSIRegressionDataset(self.test_files["intronexon"], self.tokenizer, mode=self.mode)

            # self.train_set = {
            #     "5p": PSIRegressionDataset(self.train_files["5p"], self.tokenizer),
            #     "3p": PSIRegressionDataset(self.train_files["3p"], self.tokenizer),
            #     "exon": PSIRegressionDataset(self.train_files["exon"], self.tokenizer)
            # }
            # self.val_set = {
            #     "5p": PSIRegressionDataset(self.val_files["5p"], self.tokenizer),
            #     "3p": PSIRegressionDataset(self.val_files["3p"], self.tokenizer),
            #     "exon": PSIRegressionDataset(self.val_files["exon"], self.tokenizer)
            # }
            # self.test_set = {
            #     "5p": PSIRegressionDataset(self.test_files["5p"], self.tokenizer),
            #     "3p": PSIRegressionDataset(self.test_files["3p"], self.tokenizer),
            #     "exon": PSIRegressionDataset(self.test_files["exon"], self.tokenizer)
            # }
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
   
    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True,
            generator=torch.Generator().manual_seed(42) ###❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌ remove the seed
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True
        )


    # def _combined_loader(self, datasets):
    #     return zip(
    #         DataLoader(datasets["5p"], batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True),
    #         DataLoader(datasets["3p"], batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True),
    #         DataLoader(datasets["exon"], batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True),
    #     )

    # def train_dataloader(self):
    #     return self._combined_loader(self.train_set) if self.mode == "intronexon" else DataLoader(
    #         self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True
    #     )

    # def val_dataloader(self):
    #     return self._combined_loader(self.val_set) if self.mode == "intronexon" else DataLoader(
    #         self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True
    #     )

    # def test_dataloader(self):
    #     return self._combined_loader(self.test_set) if self.mode == "intronexon" else DataLoader(
    #         self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True
    #     )


"""
        self.train_file = config.dataset.train_file
        self.val_file = config.dataset.val_file
        self.test_file = config.dataset.test_file
        self.batch_size = config.dataset.batch_size_per_device
        self.num_workers = config.dataset.num_workers
        self.tokenizer = hydra.utils.instantiate(config.tokenizer)
        
    def setup(self, stage=None):
        
        # Load dataset and split into train/val/test sets.
        
        self.train_set = PSIRegressionDataset(self.train_file, self.tokenizer)
        self.val_set = PSIRegressionDataset(self.val_file, self.tokenizer)
        self.test_set = PSIRegressionDataset(self.test_file, self.tokenizer)
    
    
    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )


"""
        