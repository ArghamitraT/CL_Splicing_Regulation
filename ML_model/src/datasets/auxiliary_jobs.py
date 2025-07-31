import torch
import hydra
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
import pickle
import lightning.pytorch as pl


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

        # Fixed lengths for MTSplice windowing
        self.len_5p = 200
        self.len_exon = 100
        self.len_3p = 200
        self.tissue_acceptor_intron = 300
        self.tissue_acceptor_exon = 100
        self.tissue_donor_intron = 300
        self.tissue_donor_exon = 100

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):

        exon_id, entry = self.entries[idx]
        psi_value = entry["psi_val"]

        if self.mode == "mtsplice":
            full_seq =  entry["5p"] + self._process_exon(entry["exon"]) + entry["3p"]

            windows = get_windows_with_padding(
                full_seq,
                self.len_5p, self.len_exon, self.len_3p,
                self.tissue_acceptor_intron, self.tissue_acceptor_exon,
                self.tissue_donor_exon, self.tissue_donor_intron
            )

            # Tokenize acceptor and donor
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

        # elif self.mode == "intronexon":
        else:
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
        # else:
        #     raise ValueError(f"Unsupported mode: {self.mode}")
   
    def train_dataloader(self):
        # return DataLoader(
        #     self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True,
        #     generator=torch.Generator().manual_seed(42) ###❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌ remove the seed
        # )
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

        