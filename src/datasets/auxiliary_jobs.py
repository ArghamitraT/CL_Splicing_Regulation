import torch
import hydra
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
import pickle
import lightning.pytorch as pl

from .utility import (
    get_windows_with_padding
)

class PSIRegressionDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=201, mode="5p", len_5p=300, len_3p=300, ascot=False):
        """
        Dataset for PSI Regression.

        Args:
            data_file (str): Path to the pickle file containing PSI values and sequences.
            tokenizer_name (str): Name of the tokenizer to use.
            max_length (int): Max sequence length for padding.
        """
        
        self.tokenizer = tokenizer

        # Load data from pickle file
        with open(data_file, "rb") as f:
            self.data = pickle.load(f)

        self.max_length = max_length
        self.mode = mode
        self.entries = list(self.data.items())  
        self.len_5p = len_5p
        self.len_exon = 100
        self.len_3p = len_3p
        self.tissue_acceptor_intron = 300
        self.tissue_donor_intron = 300
        self.tissue_acceptor_exon = 100
        self.tissue_donor_exon = 100
        self.ascot = ascot

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):

        exon_id, entry = self.entries[idx]
        psi_value = entry["psi_val"]

        if self.mode == "mtsplice":
        
            if self.len_3p == 200:
                entry["5p"] = entry["5p"][-200:]
                entry["3p"] = entry["3p"][:200]
                
            full_seq =  entry["5p"] + self._process_exon(entry["exon"]) + entry["3p"]
            
            windows = get_windows_with_padding(self.tissue_acceptor_intron, self.tissue_donor_intron, self.tissue_acceptor_exon, self.tissue_donor_exon, full_seq, overhang = (self.len_3p, self.len_5p))
            
            seql = self._tokenize(windows['acceptor'])  # acceptor
            seqr = self._tokenize(windows['donor'])     # donor

            return (seql, seqr), torch.tensor(psi_value, dtype=torch.float32), exon_id


        elif self.mode == "intronexon" or self.mode == "intronOnly":
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
        return start+end

        

    def _tokenize(self, seq):
        if callable(self.tokenizer) and not hasattr(self.tokenizer, "vocab_size"):
            tokenized = self.tokenizer([seq])
            return tokenized[0]
        else:
            return self.tokenizer(
                seq,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            ).input_ids.squeeze(0)

        
class PSIRegressionDataModule(pl.LightningDataModule):
    def __init__(self, config):
        """
        PyTorch Lightning DataModule for PSI Regression.

        Args:
            config (OmegaConf): Config object with dataset parameters.
        """
        super().__init__()
        self.config = config
        self.mode = config.aux_models.mode 

        self.batch_size = config.dataset.batch_size_per_device
        self.num_workers = config.dataset.num_workers
        self.tokenizer = hydra.utils.instantiate(config.tokenizer)

        self.train_files = config.dataset.train_files
        self.val_files = config.dataset.val_files
        self.test_files = config.dataset.test_files
        self.len_5p = config.dataset.fivep_ovrhang  # how much overhang to include from 5' intron
        self.len_3p = config.dataset.threep_ovrhang # how much overhang to include from 3' intron
        self.ascot = config.dataset.ascot

    def setup(self, stage=None):
        
        self.train_set = PSIRegressionDataset(self.train_files["intronexon"], self.tokenizer, mode=self.mode,len_5p=self.len_5p, len_3p=self.len_3p, ascot=self.ascot)
        self.val_set = PSIRegressionDataset(self.val_files["intronexon"], self.tokenizer, mode=self.mode,len_5p=self.len_5p, len_3p=self.len_3p, ascot=self.ascot)
        self.test_set = PSIRegressionDataset(self.test_files["intronexon"], self.tokenizer, mode=self.mode,len_5p=self.len_5p, len_3p=self.len_3p, ascot=self.ascot)

   
    def train_dataloader(self):
        
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True
        )

        
