"""
Gets exon intron position from .zarr and gencode annotation
"""

import pandas as pd

# Step 1: Load the GTF file
gtf_file = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/initial_data/gencode.v47.basic.annotation.gtf"
# gtf_file = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/initial_data/dummy.gtf"

# Define column names for the GTF file
gtf_columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']

# Read the GTF file into a pandas DataFrame
gtf_df = pd.read_csv(gtf_file, sep='\t', comment='#', header=None, names=gtf_columns)

# Step 2: Filter for 'exon' feature rows and explicitly make a copy to avoid SettingWithCopyWarning
exon_df = gtf_df[gtf_df['feature'] == 'exon'].copy()

# Function to extract exon ID from the attribute column
def extract_exon_id(attribute_string):
    for attribute in attribute_string.split(';'):
        if 'exon_id' in attribute or 'transcript_id' in attribute:  # depending on the id naming convention in the file
            return attribute.split('"')[1]  # return the value inside quotes
    return None

# Apply the function to extract exon id using .loc to avoid the warning
exon_df.loc[:, 'exon_id'] = exon_df['attribute'].apply(extract_exon_id)

def calculate_intron_upstream(row):
    try:
        # Ensure that the start and end values are integers
        start = int(row['start'])
        end = int(row['end'])

        if row['strand'] == '+':
            # Check if the exon start is large enough to calculate upstream
            if start > 200:
                intron_start = start - 200
                intron_end = start - 1
            else:
                intron_start, intron_end = None, None  # Invalid upstream case
        else:
            # For the negative strand, calculate the downstream (as upstream is after the exon end)
            intron_start = end + 1
            intron_end = end + 200

        # Ensure the results are either valid integers or None
        return pd.Series([intron_start, intron_end], index=['intron_start', 'intron_end'])
    
    except (ValueError, TypeError):
        # In case there is some data issue, return None for both intron_start and intron_end
        return pd.Series([None, None], index=['intron_start', 'intron_end'])

# Apply the function to calculate upstream introns using .loc
exon_df.loc[:, ['intron_start', 'intron_end']] = exon_df.apply(calculate_intron_upstream, axis=1)


# Step 4: Select columns of interest and save to CSV
output_df = exon_df[['seqname', 'start', 'end', 'strand', 'exon_id', 'intron_start', 'intron_end']]
output_df.columns = ['chromosome', 'exon_start', 'exon_end', 'strand', 'exon_id', 'upstream_intron_start', 'upstream_intron_end']

# Save the result to a CSV file
output_df.to_csv("/gpfs/commons/home/atalukder/Contrastive_Learning/data/initial_data/exon_introns_with_200bp_upstream.csv", index=False)

print(output_df.head())