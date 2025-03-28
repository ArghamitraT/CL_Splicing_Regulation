import pandas as pd
import numpy as np

# Load the GTF file, filtering for exons only
print("Loading GTF file and extracting exon information...")
gtf_file_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/models/gene_exp_psi/Homo_sapiens.GRCh38.91.gtf'
gtf_df = pd.read_csv(gtf_file_path, sep='\t', header=None, comment='#')
# gtf_df = gtf_df[gtf_df[2] == 'exon'].head(10000)  # (AT)

# Get exon names, transcript, and gene IDs more efficiently
gtf_df['exon_name'] = gtf_df[0].astype(str) + '|' + gtf_df[3].astype(str) + '|' + gtf_df[4].astype(str) + '|' + gtf_df[6].astype(str)
gtf_df['transcript_id'] = gtf_df[8].str.extract(r'transcript_id "([^"]+)"')[0]
gtf_df['gene_id'] = gtf_df[8].str.extract(r'gene_id "([^"]+)"')[0]

# Drop unnecessary columns to save memory
gtf_df = gtf_df[['exon_name', 'transcript_id', 'gene_id']]

# Compute unique signatures for exons directly with groupby and join
print("Computing unique signatures for exons...")
# unique_exon_signatures = gtf_df.groupby('exon_name')['transcript_id'].apply(lambda x: '-'.join(x.unique())).to_dict()
unique_exon_signatures = gtf_df.groupby('exon_name')['transcript_id'].apply(
    lambda x: '-'.join(x.dropna().unique().astype(str))
).to_dict()         # for each exon it joins the transcript names with '-'

# Filter out exons of genes that have only one transcript
print("Filtering exons by transcript count...")
gene_transcript_counts = gtf_df.groupby('gene_id')['transcript_id'].nunique()   # for each gene finds out the transcript number
multi_transcript_genes = gene_transcript_counts[gene_transcript_counts > 1].index   # which genes have more than 1 transcripts
filtered_exon_signatures = {exon: sig for exon, sig in unique_exon_signatures.items() if gtf_df[gtf_df['exon_name'] == exon]['gene_id'].iloc[0] in multi_transcript_genes}      #contains only those exons where the associated gene has more than one transcript.

# Create a table that maps transcripts to exons
transcript_to_exon = gtf_df.copy()

# Count the number of transcripts of each gene, transcripts per exon, and exons per transcript
print("Counting transcripts and exons...")
transcript_to_exon['gene_transcript_count'] = transcript_to_exon['gene_id'].map(gene_transcript_counts)     # Number of unique transcripts for each gene.
transcript_to_exon['exon_transcript_count'] = transcript_to_exon.groupby('exon_name')['transcript_id'].transform('nunique')     # eg. exon1 is found in both tx1 and tx2, so exon_transcript_count is 2 for both rows with exon1.
transcript_to_exon['transcript_exon_count'] = transcript_to_exon.groupby('transcript_id')['exon_name'].transform('nunique')     # Number of unique exons within each transcript.

# Apply filters in a vectorized way
print("Applying filters to select relevant transcripts and exons...")
transcript_to_exon_filtered = transcript_to_exon[
    (transcript_to_exon['exon_transcript_count'] < transcript_to_exon['gene_transcript_count']) &
    (transcript_to_exon['gene_transcript_count'] > 1) &
    (transcript_to_exon['transcript_exon_count'] > 1)
]
transcript_to_exon_filtered = transcript_to_exon_filtered.drop(columns=['gene_transcript_count', 'exon_transcript_count', 'transcript_exon_count'])

# Keep only one exon for each unique signature
print("Selecting one exon per unique signature...")
unique_exon_dict = {}
for exon, sig in filtered_exon_signatures.items():
    if sig not in unique_exon_dict:
        unique_exon_dict[sig] = exon  # Keep the first exon for each unique signature

# Filter transcript_to_exon_filtered to only include these unique exons
unique_exons = set(unique_exon_dict.values())
transcript_to_exon_final = transcript_to_exon_filtered[transcript_to_exon_filtered['exon_name'].isin(unique_exons)]

# Save the final output
output_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/models/gene_exp_psi/output/after_exon_sig_next.pkl'
# transcript_to_exon_final.to_pickle(output_path) #(AT)
print(f"Saved final output to {output_path}")
