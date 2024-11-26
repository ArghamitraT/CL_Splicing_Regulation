import torch
import hydra
from transformers import AutoTokenizer
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split
from src.datasets.introns_alignment import ContrastiveIntronsDataset
from src.datasets.constitutive_introns import ConstitutiveIntronsDataset
from src.datasets.psi_lung_introns import PsiLungIntronsDataset

class ContrastiveIntronsDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.data_file = config.dataset.data_file
        self.exon_names_path = config.dataset.exon_names_path
        self.batch_size = config.dataset.batch_size_per_device
        self.num_workers = config.dataset.num_workers
        self.train_ratio = config.dataset.train_ratio
        self.val_ratio = config.dataset.val_ratio
        self.test_ratio = config.dataset.test_ratio
        self.tokenizer = hydra.utils.instantiate(config.tokenizer)
        self.padding_strategy = config.tokenizer.padding

    def prepare_data(self):
        # Data preparation steps if needed, such as data checks or downloads.
        pass

    def setup(self, stage=None):
        # Create the full dataset
        full_dataset = ContrastiveIntronsDataset(
            data_file=self.data_file,
            exon_names_path=self.exon_names_path)
        
        # Determine sizes for train, validation, and test sets
        dataset_size = len(full_dataset)
        train_size = int(self.train_ratio * dataset_size)
        val_size = int(self.val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        print(f"Loaded a dataset of a total of : {dataset_size} exons, with {train_size} stored for training, {val_size} stored for validation, and {test_size} stored for testing.")

        # Split dataset into train, validation, and test
        self.train_set, self.val_set, self.test_set = random_split(
            full_dataset,
            [train_size, val_size, test_size]
        )  
    
    def collate_fn(self, batch):
        # Separate all augmentations from batch into two lists
        view1_sequences = [item[0] for item in batch]
        view2_sequences = [item[1] for item in batch]
        
        # Tokenize both views in batch mode
        view1 = self.tokenizer(
            view1_sequences, 
            return_tensors='pt',
            padding=self.padding_strategy,
        ).input_ids
        view2 = self.tokenizer(
            view2_sequences, 
            return_tensors='pt',
            padding=self.padding_strategy,
        ).input_ids
        
        return view1, view2      
        
    def train_dataloader(self):
        return DataLoader(
            self.train_set, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            collate_fn=self.collate_fn, 
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            collate_fn=self.collate_fn, 
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            collate_fn=self.collate_fn, 
            pin_memory=True
        )
        

class ConstitutiveIntronsDataModule(pl.LightningDataModule):
    def __init__(self, csv_file, tokenizer, padding_strategy='longest', batch_size=32, num_workers=4):
        """
        Args:
            csv_file (str): Path to the .csv file containing the dataset.
            tokenizer: HuggingFace tokenizer instance for tokenizing sequences.
            padding_strategy (str): Padding strategy for tokenization ('longest', 'max_length', etc.).
            seq_length (int): Optional fixed sequence length for truncation or padding.
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of workers for DataLoader.
        """
        super().__init__()
        self.csv_file = csv_file
        self.tokenizer = tokenizer
        self.padding_strategy = padding_strategy
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """
        Called on setup to prepare the datasets for train/val/test splits.
        """
        # Load the dataset
        full_dataset = ConstitutiveIntronsDataset(self.csv_file)

        # Split dataset into train, val, and test sets (70/15/15 split)
        train_size = int(0.8 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def collate_fn(self, batch):
        """
        Custom collate function to tokenize sequences and prepare batches.
        """
        seq, label = zip(*batch)  # Unpack the batch into sequences and labels

        # Tokenize the sequences
        tokenized_sequences = self.tokenizer(
            list(seq),  # Tokenizer expects a list of sequences
            return_tensors='pt',
            padding=self.padding_strategy,
        ).input_ids

        # Convert labels to tensor
        labels = torch.tensor(label, dtype=torch.long)

        return tokenized_sequences, labels

    def train_dataloader(self):
        """
        Returns DataLoader for the training set.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          shuffle=True, num_workers=self.num_workers, 
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        """
        Returns DataLoader for the validation set.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers, 
                          collate_fn=self.collate_fn)

    def test_dataloader(self):
        """
        Returns DataLoader for the test set.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers, 
                          collate_fn=self.collate_fn)

class PsiLungIntronsDataModule(pl.LightningDataModule):
    def __init__(self, csv_file, tokenizer, padding_strategy='longest', batch_size=32, num_workers=4):
        """
        Args:
            csv_file (str): Path to the .csv file containing the dataset.
            tokenizer: HuggingFace tokenizer instance for tokenizing sequences.
            padding_strategy (str): Padding strategy for tokenization ('longest', 'max_length', etc.).
            seq_length (int): Optional fixed sequence length for truncation or padding.
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of workers for DataLoader.
        """
        super().__init__()
        self.csv_file = csv_file
        self.tokenizer = tokenizer
        self.padding_strategy = padding_strategy
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """
        Called on setup to prepare the datasets for train/val/test splits.
        """
        # Load the dataset
        full_dataset = PsiLungIntronsDataset(self.csv_file)

        # Split dataset into train, val, and test sets (70/15/15 split)
        train_size = int(0.8 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def collate_fn(self, batch):
        """
        Custom collate function to tokenize sequences and prepare batches.
        """
        seq, label = zip(*batch)  # Unpack the batch into sequences and labels

        # Tokenize the sequences
        tokenized_sequences = self.tokenizer(
            list(seq),  # Tokenizer expects a list of sequences
            return_tensors='pt',
            padding=self.padding_strategy,
        ).input_ids

        # Convert labels to tensor
        labels = torch.tensor(label, dtype=torch.float32)

        return tokenized_sequences, labels

    def train_dataloader(self):
        """
        Returns DataLoader for the training set.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          shuffle=True, num_workers=self.num_workers, 
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        """
        Returns DataLoader for the validation set.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers, 
                          collate_fn=self.collate_fn)

    def test_dataloader(self):
        """
        Returns DataLoader for the test set.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=self.num_workers, 
                          collate_fn=self.collate_fn)