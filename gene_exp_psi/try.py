import pandas as pd
import numpy as np
# from sklearn.linear_model import LinearRegression
# from scipy.stats import linregress
import pickle


# print("Loading GTEx counts...")
# gtex_counts = pd.read_csv('/gpfs/commons/home/atalukder/Contrastive_Learning/models/gene_exp_psi/GTEx_Analysis_2017-06-05_v8_RSEMv1.3.0_transcript_tpm.gct', sep='\t', skiprows=2)
# print(gtex_counts.head(10))

# gtex_counts = gtex_counts.head(100000)
# gtex_counts.to_pickle('/gpfs/commons/home/atalukder/Contrastive_Learning/models/gene_exp_psi/GTEx_Analysis_2017-06-05_v8_RSEMv1.3.0_transcript_tpm_short.gct')print("Loading GTEx counts...")

tissues = [
    'Lung', 'Spleen', 'Thyroid', 'Brain - Cortex', 'Adrenal Gland',
    'Breast - Mammary Tissue', 'Heart - Left Ventricle', 'Liver', 'Pituitary', 'Pancreas'
]
tissue_index = 0  # Example index, change as needed

gtex_counts = pd.read_csv('/gpfs/commons/home/atalukder/Contrastive_Learning/models/gene_exp_psi/GTEx_Analysis_2017-06-05_v8_RSEMv1.3.0_transcript_tpm.gct', sep='\t', skiprows=2)
# gtex_counts = pd.read_csv('/gpfs/commons/home/atalukder/Contrastive_Learning/models/gene_exp_psi/GTEx_Analysis_2017-06-05_v8_RSEMv1.3.0_transcript_tpm.gct', sep='\t', skiprows=2)
meta_data = pd.read_csv('/gpfs/commons/home/atalukder/Contrastive_Learning/models/gene_exp_psi/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt', sep='\t')

# Filter metadata for the specified tissue
meta_data = meta_data[meta_data['SMTSD'] == tissues[tissue_index]]
sample_ids = meta_data['SAMPID'].values
gtex_counts = gtex_counts[['transcript_id', 'gene_id'] + list(sample_ids)]
print("done")

#