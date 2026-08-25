import pandas as pd
import pickle

file = "/gpfs/commons/home/nkeung/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/test_3primeIntron_filtered.pkl"

df = pickle.load(open(f'{file}', 'rb'))
exon = list(df)[0]
print(df[exon])
