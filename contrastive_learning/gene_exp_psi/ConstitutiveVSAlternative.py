"""
from the csv anntation files (exactly like gtf but in csv format) it lables exons constitutive or not
"""

import pandas as pd
import pickle

# Define the function to calculate intron start and end positions
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

# Load the exon-transcript mapping data
print("Loading exon-transcript mapping data...")
# with open('/gpfs/commons/home/atalukder/Contrastive_Learning/data/fine_tuning/gene_exp_psi/after_exon_sig_next.pkl', 'rb') as f:
#     transcript_to_exon = pickle.load(f)
csv_file_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/fine_tuning/gene_exp_psi/Homo_sapiens.GRCh38.91.csv'
print("Loading exon-transcript mapping data from CSV...")
transcript_to_exon = pd.read_csv(csv_file_path)

# Calculate the number of unique transcripts for each gene-exon pair
gene_exon_transcript_counts = transcript_to_exon.groupby(['gene_id', 'exon_name'])['transcript_id'].nunique()
transcript_to_exon = transcript_to_exon.merge(gene_exon_transcript_counts.rename('gene_exon_transcript_count'), on=['gene_id', 'exon_name'])

# Calculate the total transcript count for each gene
gene_transcript_counts = transcript_to_exon.groupby('gene_id')['transcript_id'].nunique()
transcript_to_exon['gene_transcript_count'] = transcript_to_exon['gene_id'].map(gene_transcript_counts)

# Define constitutive exon based on gene-specific counts
transcript_to_exon['constitutive_exon'] = (transcript_to_exon['gene_exon_transcript_count'] == transcript_to_exon['gene_transcript_count']).astype(int)

# Extract exons present in all transcripts of their gene (constitutive) and not (alternative)
exons_in_all_transcripts = transcript_to_exon[transcript_to_exon['constitutive_exon'] == 1].drop_duplicates(subset=['exon_name', 'gene_id'])
exons_not_in_all_transcripts = transcript_to_exon[transcript_to_exon['constitutive_exon'] == 0].drop_duplicates(subset=['exon_name', 'gene_id'])

# Combine both sets of exons
all_exons = pd.concat([exons_in_all_transcripts, exons_not_in_all_transcripts])

# Ensure chromosome values have "chr" prefix and extract other values as needed
all_exons['Chromosome'] = all_exons['exon_name'].apply(lambda x: "chr" + x.split('|')[0])
all_exons['Exon Start'] = all_exons['exon_name'].apply(lambda x: int(x.split('|')[1]))
all_exons['Exon End'] = all_exons['exon_name'].apply(lambda x: int(x.split('|')[2]))
all_exons['Strand'] = all_exons['exon_name'].apply(lambda x: x.split('|')[3])

# Calculate intron start and end using the calculate_intron function
all_exons[['Intron Start', 'Intron End']] = all_exons.apply(
    lambda row: calculate_intron(row['Exon Start'], row['Exon End'], row['Strand']), axis=1, result_type='expand'
)

# Add Species Name column with "hg38"
all_exons['Species Name'] = 'hg38'

# Organize columns to match the desired format
formatted_df = all_exons[['exon_name', 'Species Name', 'Chromosome', 'Exon Start', 'Exon End', 'Strand', 'Intron Start', 'Intron End', 'constitutive_exon']]
formatted_df.rename(columns={'exon_name': 'Exon Name', 'constitutive_exon': 'Constitutive Exon'}, inplace=True)

# Save to CSV
output_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/fine_tuning/exons_with_introns.csv'
formatted_df.to_csv(output_path, index=False)

print(f"CSV file saved to {output_path}")














# import pandas as pd
# import pickle

# # Define the function to calculate intron start and end positions
# def calculate_intron(start, end, strand):
#     if strand == '+':
#         if start > 200:
#             intron_start = start - 200
#             intron_end = start - 1
#         else:
#             intron_start, intron_end = None, None  # Invalid upstream case
#     else:
#         intron_start = end + 1
#         intron_end = end + 200
#     return intron_start, intron_end

# # Load the exon-transcript mapping data
# print("Loading exon-transcript mapping data...")
# with open('/gpfs/commons/home/atalukder/Contrastive_Learning/data/fine_tuning/gene_exp_psi/after_exon_sig_next.pkl', 'rb') as f:
#     transcript_to_exon = pickle.load(f)

# # Calculate the number of unique transcripts for each gene (gene_transcript_count)
# gene_transcript_counts = transcript_to_exon.groupby('gene_id')['transcript_id'].nunique()
# transcript_to_exon['gene_transcript_count'] = transcript_to_exon['gene_id'].map(gene_transcript_counts)

# # Calculate the number of unique transcripts each exon is associated with (exon_transcript_count)
# exon_transcript_counts = transcript_to_exon.groupby('exon_name')['transcript_id'].nunique()
# transcript_to_exon['exon_transcript_count'] = transcript_to_exon['exon_name'].map(exon_transcript_counts)

# # Identify exons present in all transcripts of their gene
# exons_in_all_transcripts = transcript_to_exon[
#     transcript_to_exon['exon_transcript_count'] == transcript_to_exon['gene_transcript_count']
# ].drop_duplicates(subset=['exon_name', 'gene_id'])

# # Identify exons not present in all transcripts of their gene
# exons_not_in_all_transcripts = transcript_to_exon[
#     transcript_to_exon['exon_transcript_count'] < transcript_to_exon['gene_transcript_count']
# ].drop_duplicates(subset=['exon_name', 'gene_id'])

# # Combine both sets of exons
# all_exons = pd.concat([exons_in_all_transcripts, exons_not_in_all_transcripts])

# # Assuming the 'exon_name' column has the chromosome, start, end, and strand information in the required format
# # If these values are available in other columns, replace accordingly
# all_exons['Chromosome'] = all_exons['exon_name'].apply(lambda x: "chr" + x.split('|')[0])  # Add "chr" prefix to chromosome
# all_exons['Exon Start'] = all_exons['exon_name'].apply(lambda x: int(x.split('|')[1]))
# all_exons['Exon End'] = all_exons['exon_name'].apply(lambda x: int(x.split('|')[2]))
# all_exons['Strand'] = all_exons['exon_name'].apply(lambda x: x.split('|')[3])

# # Calculate intron start and end using the calculate_intron function
# all_exons[['Intron Start', 'Intron End']] = all_exons.apply(
#     lambda row: calculate_intron(row['Exon Start'], row['Exon End'], row['Strand']), axis=1, result_type='expand'
# )

# # Add the Species Name column with "hg38"
# all_exons['Species Name'] = 'hg38'

# # Organize columns to match the desired format
# formatted_df = all_exons[['exon_name', 'Species Name', 'Chromosome', 'Exon Start', 'Exon End', 'Strand', 'Intron Start', 'Intron End']]
# formatted_df.rename(columns={'exon_name': 'Exon Name'}, inplace=True)

# # Save to CSV
# output_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/fine_tuning/exons_with_introns.csv'
# formatted_df.to_csv(output_path, index=False)

# print(f"CSV file saved to {output_path}")
