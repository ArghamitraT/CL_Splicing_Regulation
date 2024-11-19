"""
Loops through all tissues
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
import pickle

# Define tissue names
tissues = [
    'Lung', 'Spleen', 'Thyroid', 'Brain - Cortex', 'Adrenal Gland',
    'Breast - Mammary Tissue', 'Heart - Left Ventricle', 'Liver', 'Pituitary', 'Pancreas'
]

gene_psi_data_folder ='/gpfs/commons/home/atalukder/Contrastive_Learning/data/fine_tuning/gene_exp_psi/'
# Load preprocessed exon-transcript mapping
print("Loading exon-transcript mapping data...")
with open(gene_psi_data_folder+'after_exon_sig_next.pkl', 'rb') as f:
    transcript_to_exon = pickle.load(f)

# Load GTF file to get exon names, transcript IDs, and gene IDs
print("Loading GTF file...")
gtf_file = pd.read_csv(gene_psi_data_folder+'Homo_sapiens.GRCh38.91.gtf', sep='\t', header=None, comment='#')
gtf_file = gtf_file[gtf_file[2] == 'exon']  # Filter for exon rows
gtf_file.columns = ["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attributes"]

# Extract exon names, transcript IDs, and gene IDs
gtf_file['exon_name'] = gtf_file['chrom'].astype(str) + '|' + gtf_file['start'].astype(str) + '|' + \
                        gtf_file['end'].astype(str) + '|' + gtf_file['strand'].astype(str)
gtf_file['transcript_id'] = gtf_file['attributes'].str.extract(r'transcript_id "([^"]+)"')[0]
gtf_file['gene_id'] = gtf_file['attributes'].str.extract(r'gene_id "([^"]+)"')[0]

# Load GTEx counts once, as it is the same for all tissues
print("Loading GTEx counts...")
gtex_counts = pd.read_csv(gene_psi_data_folder+'GTEx_Analysis_2017-06-05_v8_RSEMv1.3.0_transcript_tpm.gct', sep='\t', skiprows=2)
meta_data = pd.read_csv(gene_psi_data_folder+'GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt', sep='\t')

# Loop through each tissue and perform the analysis
for tissue_index, tissue in enumerate(tissues):
    print(f"Processing tissue: {tissue}")

    # Filter metadata for the current tissue
    tissue_meta_data = meta_data[meta_data['SMTSD'] == tissue]
    sample_ids = tissue_meta_data['SAMPID'].values

    # Filter sample IDs to keep only those that exist in gtex_counts
    sample_ids = [samp_id for samp_id in sample_ids if samp_id in gtex_counts.columns]
    tissue_meta_data = tissue_meta_data[tissue_meta_data['SAMPID'].isin(sample_ids)]

    # Subset gtex_counts to include only the relevant sample IDs and the necessary columns
    tissue_gtex_counts = gtex_counts[['transcript_id', 'gene_id'] + sample_ids]

    # Clean up transcript and gene IDs
    tissue_gtex_counts['transcript_id'] = tissue_gtex_counts['transcript_id'].str.replace(r'\.[0-9]+', '', regex=True)
    tissue_gtex_counts['gene_id'] = tissue_gtex_counts['gene_id'].str.replace(r'\.[0-9]+', '', regex=True)

    # Calculate exon proportions and counts
    print("Calculating exon proportions and counts...")
    exon_proportions = []
    exon_gene_counts = []

    for exon in transcript_to_exon['exon_name'].unique():
        exon_data = transcript_to_exon[transcript_to_exon['exon_name'] == exon]
        gene = exon_data['gene_id'].values[0]
        transcripts_in_exon = exon_data['transcript_id'].unique()
        all_gene_transcripts = tissue_gtex_counts[tissue_gtex_counts['gene_id'] == gene]['transcript_id'].unique()
        
        # Skip if there are no gene transcripts
        # if len(all_gene_transcripts) == 0:
        #     continue
        
        # Sum counts across transcripts for each exon
        total_counts = tissue_gtex_counts[tissue_gtex_counts['transcript_id'].isin(all_gene_transcripts)].iloc[:, 2:].sum(axis=0)
        exon_gene_counts.append(total_counts)

        # Calculate proportions
        exon_counts = tissue_gtex_counts[tissue_gtex_counts['transcript_id'].isin(transcripts_in_exon)].iloc[:, 2:].sum(axis=0)
        proportions = exon_counts / total_counts
        proportions.replace([np.inf, -np.inf], np.nan, inplace=True)
        exon_proportions.append(proportions)

    # Convert to DataFrames
    exon_proportions_df = pd.DataFrame(exon_proportions, index=transcript_to_exon['exon_name'].unique())
    exon_gene_counts_df = pd.DataFrame(exon_gene_counts, index=transcript_to_exon['exon_name'].unique())

    # Compute correlations between rows of the two matrices
    print("Calculating correlations...")
    cor_vals = []
    for i in range(len(exon_proportions_df)):
        proportions = exon_proportions_df.iloc[i, :].values
        gene_counts = exon_gene_counts_df.iloc[i, :].values
        
        if np.isnan(proportions).sum() >= len(proportions) / 2:
            cor_vals.append([np.nan, np.nan, np.nan])
            continue
        
        if np.nanquantile(gene_counts, 0.95) / np.nanquantile(gene_counts, 0.05) < 2:
            cor_vals.append([np.nan, np.nan, np.nan])
            continue
        
        if np.nanstd(proportions) == 0 or np.nanstd(gene_counts) == 0:
            cor_vals.append([np.nan, np.nan, np.nan])
            continue
        
        # Filter out NaN values for alignment
        valid_mask = ~np.isnan(proportions) & ~np.isnan(gene_counts)
        filtered_proportions = proportions[valid_mask]
        filtered_gene_counts = gene_counts[valid_mask]
        
        # Perform linear regression
        reg_result = linregress(filtered_proportions, filtered_gene_counts)
        cor_vals.append([reg_result.slope, reg_result.pvalue, reg_result.rvalue ** 2])

    # Convert correlation results to DataFrame
    cor_vals_df = pd.DataFrame(cor_vals, columns=['coef', 'pval', 'r_squared'])
    print(f"Correlation calculations completed for {tissue}.")

    # Save output for the current tissue
    output_file = gene_psi_data_folder+f'output/types_{tissue}.pkl'
    cor_vals_df.to_pickle(output_file)
    print(f"Results saved to {output_file} for {tissue}.")
