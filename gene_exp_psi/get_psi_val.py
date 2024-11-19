"""
For each exon gets these values:
Exon Name,Species Name,Chromosome,Exon Start,Exon End,Strand,Intron Start,Intron End and psi value
"""


import pandas as pd
import numpy as np
import pickle
from scipy.stats import linregress

# Function to extract information from exon_name
def parse_exon_name(exon_name):
    parts = exon_name.split('|')
    chromosome = 'chr'+parts[0]
    exon_start = int(parts[1])
    exon_end = int(parts[2])
    strand = parts[3]
    return chromosome, exon_start, exon_end, strand

# Function to calculate intron start and end
def calculate_intron(start, end, strand):
    if strand == '+':
        if start > 200:
            intron_start = start - 200
            intron_end = start - 1
        else:
            intron_start, intron_end = None, None  # Invalid upstream case
    else:
        intron_start = end + 1
        intron_end = end + 200
    return intron_start, intron_end

# Define paths
data_folder = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/fine_tuning/gene_exp_psi/'
path_to_gtf = data_folder + 'Homo_sapiens.GRCh38.91.gtf'
path_to_gtex_counts = data_folder + 'GTEx_Analysis_2017-06-05_v8_RSEMv1.3.0_transcript_tpm.gct'
path_to_gtex_metadata = data_folder + 'GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt'
output_folder = data_folder + 'output_with_sig/'  # Define output folder

# Load preprocessed exon-transcript mapping from `after_exon_sig_next.pkl`
print("Loading exon-transcript mapping data...")
with open(data_folder + 'after_exon_sig_next.pkl', 'rb') as f:
    transcript_to_exon = pickle.load(f)

# Load GTEx counts and metadata
gtex_counts = pd.read_csv(path_to_gtex_counts, sep='\t', skiprows=2) ##(AT)
# gtex_counts = pd.read_pickle(data_folder + 'GTEx_Analysis_2017-06-05_v8_RSEMv1.3.0_transcript_tpm_short.gct.pkl')
meta_data = pd.read_csv(path_to_gtex_metadata, sep='\t')

# Define tissues (AT)
# tissues = [
#     'Lung', 'Spleen', 'Thyroid', 'Brain - Cortex', 'Adrenal Gland',
#     'Breast - Mammary Tissue', 'Heart - Left Ventricle', 'Liver', 'Pituitary', 'Pancreas'
# ]
tissues = [
    'Lung'
]

# Process each tissue
for tissue in tissues:
    print(f"Processing tissue: {tissue}")

    # Filter metadata for the current tissue
    tissue_meta_data = meta_data[meta_data['SMTSD'] == tissue]
    sample_ids = tissue_meta_data['SAMPID'].values

    # Filter sample IDs to keep only those that exist in gtex_counts
    sample_ids = [samp_id for samp_id in sample_ids if samp_id in gtex_counts.columns]
    tissue_meta_data = tissue_meta_data[tissue_meta_data['SAMPID'].isin(sample_ids)]

    # Filter GTEx counts based on sample IDs
    tissue_gtex_counts = gtex_counts[['transcript_id', 'gene_id'] + list(sample_ids)]
    tissue_gtex_counts['transcript_id'] = tissue_gtex_counts['transcript_id'].str.replace(r'\.[0-9]+', '', regex=True)
    tissue_gtex_counts['gene_id'] = tissue_gtex_counts['gene_id'].str.replace(r'\.[0-9]+', '', regex=True)

    # Initialize lists to store data
    exon_data_records = []

    # (AT)
    # Select first 1000 unique exons
    # unique_exons = transcript_to_exon['exon_name'].unique()[:1000]
    unique_exons = transcript_to_exon['exon_name'].unique()
    # Calculate exon proportions and gene counts
    for exon in unique_exons:

        # Extract chromosome, exon start, exon end, and strand from exon_name
        chromosome, exon_start, exon_end, strand = parse_exon_name(exon)

        # Calculate intron start and end
        intron_start, intron_end = calculate_intron(exon_start, exon_end, strand)

        # Skip if intron is invalid
        if intron_start is None or intron_end is None:
            continue

        exon_data = transcript_to_exon[transcript_to_exon['exon_name'] == exon]
        gene = exon_data['gene_id'].values[0]
        transcripts_in_exon = exon_data['transcript_id'].unique()
        all_gene_transcripts = tissue_gtex_counts[tissue_gtex_counts['gene_id'] == gene]['transcript_id'].unique()
        
        # Skip if there are no gene transcripts
        if len(all_gene_transcripts) == 0:
            continue
        
        # Sum counts across transcripts for each exon
        exon_counts = tissue_gtex_counts[tissue_gtex_counts['transcript_id'].isin(transcripts_in_exon)].iloc[:, 2:].sum(axis=0)
        gene_counts = tissue_gtex_counts[tissue_gtex_counts['transcript_id'].isin(all_gene_transcripts)].iloc[:, 2:].sum(axis=0)
        
        # Calculate PSI
        psi = exon_counts / gene_counts
        psi.replace([np.inf, -np.inf], np.nan, inplace=True)
        ## Store record for this exon
        exon_data_records.append({
            'Exon Name': exon,
            'Species Name':'hg38',
            'Gene ID': gene,
            'Chromosome': chromosome,
            'Exon Start': exon_start,
            'Exon End': exon_end,
            'Strand': strand,
            'Intron Start': intron_start,
            'Intron End': intron_end,
            'PSI': psi.mean()  # Average PSI value across samples
        })

    # Convert to DataFrame
    exon_df = pd.DataFrame(exon_data_records)

    # Save the DataFrame as a CSV file
    output_file = f'/gpfs/commons/home/atalukder/Contrastive_Learning/data/fine_tuning/Psi_values/types_{tissue}_psi.csv'
    exon_df.to_csv(output_file, index=False)
    print(f"Saved PSI data to {output_file}")

print("Processing complete for all tissues.")