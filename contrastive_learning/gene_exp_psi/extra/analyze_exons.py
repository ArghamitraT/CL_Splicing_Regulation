"""
Redundant
"""
import pickle
import pandas as pd
from intervaltree import IntervalTree

# Define output DataFrame
columns = ['Name', 'Type', 'Chr', 'Start', 'End', 'UpIntronStart', 'UpIntronEnd', 'DownIntronStart', 'DownIntronEnd',
           'Coef', 'Slope', 'ExonNumber', 'NumExons', 'TranscriptsContain', 'TotalTranscripts', 'GeneId', 'Tissue', 'CD1cor']
exons_tab = pd.DataFrame(columns=columns)

# Define tissues
tissues = [
    'Spleen', 'Thyroid', 'Brain - Cortex', 'Adrenal Gland', 'Breast - Mammary Tissue',
    'Heart - Left Ventricle', 'Liver', 'Pituitary', 'Pancreas'
]

# Load each tissue's .pkl data and process
for tissue in tissues:
    with open(f'{output_folder}/types_{tissue}.pkl', 'rb') as f:
        data = pickle.load(f)
    
    exon_proportions = data['exon_proportions']
    exon_gene_counts = data['exon_gene_counts']
    cor_vals = data['cor_vals']
    
    # Filter exons based on thresholds for T1 and T2
    adj_p = cor_vals['pval'] <= 0.05
    high_r_squared = cor_vals['r_squared'] >= 0.5
    high_expression = exon_gene_counts.mean(axis=1) >= 20
    selected_exons = adj_p & high_r_squared & high_expression
    
    # Create IntervalTree for merged exons to find introns
    for i in selected_exons.index:
        exon_data = exon_proportions.iloc[i]
        gene_data = exon_gene_counts.iloc[i]
        row = {
            'Name': exon_data.name,
            'Type': 1 if cor_vals.iloc[i]['coef'] > 0 else 2,
            'Coef': cor_vals.iloc[i]['coef'],
            'Slope': cor_vals.iloc[i]['coef'],
            'Tissue': tissue,
            # Additional fields can be computed similarly to the R script
        }
        exons_tab = exons_tab.append(row, ignore_index=True)

# Save combined results
exons_tab.to_csv('exons_tab.csv', sep='\t', index=False)
