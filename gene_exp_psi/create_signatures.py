import pandas as pd
import numpy as np

# Load the GTF file, filtering for exons only
print("Loading GTF file and extracting exon information...")
gtf_file_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/models/gene_exp_psi/Homo_sapiens.GRCh38.91.gtf'
gtf_df = pd.read_csv(gtf_file_path, sep='\t', header=None, comment='#')
gtf_df = gtf_df[gtf_df[2] == 'exon']  # Filter for exon rows

# gtf_df = gtf_df[gtf_df[2] == 'exon'].head(10000) #(AT)

# Get exon names, transcript, and gene IDs
exon_names = gtf_df[0].astype(str) + '|' + gtf_df[3].astype(str) + '|' + gtf_df[4].astype(str) + '|' + gtf_df[6].astype(str)
transcript_ids = gtf_df[8].str.extract(r'transcript_id "([^"]+)"')[0].str.replace(';', '', regex=False)
gene_ids = gtf_df[8].str.extract(r'gene_id "([^"]+)"')[0].str.replace(';', '', regex=False)

# Load GTEx counts
# print("Loading GTEx counts...")
# gtex_counts_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/models/gene_exp_psi/GTEx_Analysis_2017-06-05_v8_RSEMv1.3.0_transcript_tpm.gct'
# gtex_counts = pd.read_csv(gtex_counts_path, nrows=5)

# Compute unique signature for exons that belong to the same set of transcripts
print("Computing unique signatures for exons...")
unique_exon_signatures = {}
for exon in exon_names.unique():
    unique_exon_signatures[exon] = '-'.join(transcript_ids[exon_names == exon].unique())

# Filter out exons of genes that only have one transcript
print("Filtering exons by transcript count...")
filtered_exon_signatures = {}
for exon, signature in unique_exon_signatures.items():
    gene_id_for_exon = gene_ids[exon_names == exon].values
    if transcript_ids[gene_ids.isin(gene_id_for_exon)].nunique() > 1:
        filtered_exon_signatures[exon] = signature

# Create a table that maps transcripts to exons
transcript_to_exon = pd.DataFrame({
    'transcript_id': transcript_ids,
    'exon_name': exon_names,
    'gene_id': gene_ids
})

# Count the number of transcripts of each gene
print("Counting transcripts per gene...")
gene_transcript_counts = transcript_to_exon.groupby('gene_id')['transcript_id'].nunique()

# Count the number of transcripts that each exon belongs to
print("Counting transcripts per exon...")
exon_transcript_counts = transcript_to_exon.groupby('exon_name')['transcript_id'].nunique()

# Count the number of exons of each transcript
print("Counting exons per transcript...")
transcript_exon_counts = transcript_to_exon.groupby('transcript_id')['exon_name'].nunique()

# Map counts to the main DataFrame
print("Mapping counts to the main DataFrame...")
transcript_to_exon['gene_transcript_count'] = transcript_to_exon['gene_id'].map(gene_transcript_counts)
transcript_to_exon['exon_transcript_count'] = transcript_to_exon['exon_name'].map(exon_transcript_counts)
transcript_to_exon['transcript_exon_count'] = transcript_to_exon['transcript_id'].map(transcript_exon_counts)


# Apply filters using the mapped columns
print("Applying final filters to transcript-to-exon mapping...")
transcript_to_exon_filtered = transcript_to_exon[
    (transcript_to_exon['exon_transcript_count'] < transcript_to_exon['gene_transcript_count']) &
    (transcript_to_exon['gene_transcript_count'] > 1) &
    (transcript_to_exon['transcript_exon_count'] > 1)
]

# Drop temporary columns if they are no longer needed
transcript_to_exon_filtered = transcript_to_exon_filtered.drop(columns=['gene_transcript_count', 'exon_transcript_count', 'transcript_exon_count'])

# Keep only one exon for each signature
print("Reducing to one exon per signature...")
unique_exon_dict = {}
for exon, sig in filtered_exon_signatures.items():
    if sig not in unique_exon_dict:
        unique_exon_dict[sig] = exon  # Keep the first exon encountered for each unique signature

# Convert the dictionary values to a set of unique exons
unique_exons = set(unique_exon_dict.values())
transcript_to_exon_final = transcript_to_exon_filtered[transcript_to_exon_filtered['exon_name'].isin(unique_exons)]

# Save the final output
output_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/models/gene_exp_psi/output/after_exon_sig_next.pkl'
transcript_to_exon_final.to_pickle(output_path)
print(f"Saved final output to {output_path}")
