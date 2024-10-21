# from gpn.data import GenomeMSA, Tokenizer
from gpn.data import GenomeMSA
# import gpn.model
# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler
# import torch
# from transformers import AutoModel, AutoModelForMaskedLM

model_path = "songlab/gpn-msa-sapiens"
# see README in https://huggingface.co/datasets/songlab/multiz100way for faster queries
msa_path = "data/initial_data/99.zarr.zip"

genome_msa = GenomeMSA(msa_path)  # can take a minute or two
# untokenized msa
raw_msa2 = genome_msa.get_msa("17", 43125250, 43125373, strand="+", tokenize=False)

x = raw_msa2[:, 4]
x_s = b''.join(x).decode('utf-8')

# raw_msa2 = genome_msa.get_msa("1", 11869, 12227, strand="+", tokenize=False)
# print(raw_msa.shape)

## tokenized msa
msa = genome_msa.get_msa("6", 31575665, 31575793, strand="+", tokenize=True)

# msa = torch.tensor(np.expand_dims(msa, 0).astype(np.int64))

# separating human from rest of species
input_ids, aux_features = msa[:, :, 0], msa[:, :, 1:]
input_ids.shape, aux_features.shape
print()