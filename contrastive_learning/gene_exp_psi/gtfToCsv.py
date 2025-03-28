"""
Takes the gtf file from:https://ftp.ensembl.org/pub/release-91/gtf/homo_sapiens/Homo_sapiens.GRCh38.91.gtf.gz
 and produces a csv file parsing the gtf
"""

import pandas as pd

# Load the GTF file, filtering out comment lines
gtf_file_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/fine_tuning/gene_exp_psi/Homo_sapiens.GRCh38.91.gtf'
gtf_df = pd.read_csv(gtf_file_path, sep='\t', header=None, comment='#')

gtf_df['exon_name'] = gtf_df[0].astype(str) + '|' + gtf_df[3].astype(str) + '|' + gtf_df[4].astype(str) + '|' + gtf_df[6].astype(str)

# Define the columns of the GTF file as per the format
gtf_df.columns = ['chromosome', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attributes', 'exon_name']

# Filter for exon entries only
gtf_df = gtf_df[gtf_df['feature'] == 'exon']
gtf_df['chromosome'] = gtf_df['chromosome'].apply(lambda x: f"chr{x}" if not str(x).startswith("chr") else x)
# Extract relevant fields from the attributes column
gtf_df['species_name'] = 'hg38'  # Assuming species is hg38 as shown in the image
# gtf_df['exon_name'] = gtf_df['attributes'].str.extract(r'gene_name "([^"]+)"')[0] + "_" + gtf_df['attributes'].str.extract(r'exon_number "([^"]+)"')[0]
# gtf_df['exon_name'] = gtf_df[0].astype(str) + '|' + gtf_df[3].astype(str) + '|' + gtf_df[4].astype(str) + '|' + gtf_df[6].astype(str)
gtf_df['exon_start'] = gtf_df['start']
gtf_df['exon_end'] = gtf_df['end']
gtf_df['gene_id'] = gtf_df['attributes'].str.extract(r'gene_id "([^"]+)"')[0]
gtf_df['transcript_id'] = gtf_df['attributes'].str.extract(r'transcript_id "([^"]+)"')[0]
gtf_df['exon_number'] = gtf_df['attributes'].str.extract(r'exon_number "([^"]+)"')[0]
gtf_df['gene_name'] = gtf_df['attributes'].str.extract(r'gene_name "([^"]+)"')[0]
gtf_df['transcript_name'] = gtf_df['attributes'].str.extract(r'transcript_name "([^"]+)"')[0]
gtf_df['exon_id'] = gtf_df['attributes'].str.extract(r'exon_id "([^"]+)"')[0]

# Select the required columns in the desired order
formatted_df = gtf_df[['species_name', 'chromosome', 'strand', 'exon_name', 'exon_start', 'exon_end', 
                       'gene_id', 'transcript_id', 'exon_number', 'gene_name', 'transcript_name', 'exon_id']]

# Save the resulting DataFrame to a CSV file
output_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/fine_tuning/gene_exp_psi/Homo_sapiens.GRCh38.91.csv'
formatted_df.to_csv(output_path, index=False)

print(f"CSV file saved to {output_path}")
