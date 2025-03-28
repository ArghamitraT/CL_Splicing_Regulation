"""
from the csv file it can fetch the intron from 99.zarr
"""
import pandas as pd
from gpn.data import GenomeMSA
import pickle

# pickle_file_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/initial_data/msa_results_TOKEN.pkl'
# with open(pickle_file_path, 'rb') as f:
#     msa_results_list_TOKEN = pickle.load(f)

# pickle_file_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/initial_data/msa_results_unTOKEN.pkl'
# with open(pickle_file_path, 'rb') as f:
#     msa_results_list_unTOKEN = pickle.load(f)


# Load the CSV file
file_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/initial_data/exon_introns_with_200bp_upstream_withoutM.csv'  # replace with your actual path
exon_data = pd.read_csv(file_path)

# Import genome_msa or replace with the actual way you're using it
# from genome_msa import get_msa

msa_results_list_TOKEN , msa_results_list_unTOKEN= [], []
msa_path = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/initial_data/99.zarr.zip"
genome_msa = GenomeMSA(msa_path)

# Function to call genome_msa.get_msa for each row
def process_row(row):
    chromosome = row['chromosome'].replace("chr", "")  # Remove "chr" from chromosome for genome_msa input
    start = int(row['upstream_intron_start'])
    end = int(row['upstream_intron_end'])
    strand = row['strand']
    
    # Replace with actual genome_msa function call
    if chromosome != 'M':
        msa_result_untoken = genome_msa.get_msa(chromosome, start, end, strand=strand, tokenize=False)
        msa_result_Token = genome_msa.get_msa(chromosome, start, end, strand=strand, tokenize=True)
        
        return msa_result_untoken, msa_result_Token

# Loop over each row and process
for index, row in exon_data.iterrows():
    msa_result_untoken, msa_result_Token = process_row(row)
    msa_results_list_TOKEN.append(msa_result_Token)
    msa_results_list_unTOKEN.append(msa_result_untoken)

with open('/gpfs/commons/home/atalukder/Contrastive_Learning/data/initial_data/msa_results_TOKEN.pkl', 'wb') as f:
    pickle.dump(msa_results_list_TOKEN, f)

with open('/gpfs/commons/home/atalukder/Contrastive_Learning/data/initial_data/msa_results_unTOKEN.pkl', 'wb') as f:
    pickle.dump(msa_results_list_unTOKEN, f)

print("Processing complete. Results saved to msa_results.pkl.")

# print("Processing complete. Results saved to msa_results.csv.")



