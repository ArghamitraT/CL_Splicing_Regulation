"""
filters exons which have a certaing length and save those as csv file
"""

import pandas as pd
import os

# Input & output paths
csv_file_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/fine_tuning/Psi_values/types_Lung_psi.csv'  # Replace with your CSV file path
output_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/fine_tuning/Psi_values/'
output_file = os.path.join(output_path, 'types_Lung_psi_exons_50to1000bp.csv')

# Load CSV
df = pd.read_csv(csv_file_path)

# Convert start/end to integers (in case they were read as strings)
df['Exon Start'] = pd.to_numeric(df['Exon Start'], errors='coerce')
df['Exon End'] = pd.to_numeric(df['Exon End'], errors='coerce')

# Compute absolute exon length and filter
df['Exon Length'] = (df['Exon End'] - df['Exon Start']).abs()
filtered_df = df[(df['Exon Length'] > 50) & (df['Exon Length'] < 1000)]

# Save result
filtered_df.to_csv(output_file, index=False)

print(f"Filtered exons saved to: {output_file}")