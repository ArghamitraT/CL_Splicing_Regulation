import pandas as pd
from gpn.data import GenomeMSA
import pickle

# pickle_file_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/initial_data/msa_results_TOKEN.pkl'
# with open(pickle_file_path, 'rb') as f:
#     msa_results_list_TOKEN = pickle.load(f)

pickle_file_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/initial_data/msa_results_unTOKEN.pkl'
with open(pickle_file_path, 'rb') as f:
    msa_results_list_unTOKEN = pickle.load(f)

print()