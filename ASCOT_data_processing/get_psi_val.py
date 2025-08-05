import pandas as pd
import numpy as np
import os
import re


# Get full absolute path of this script
script_path = os.path.abspath(__file__)
base_dir = script_path.split("Contrastive_Learning")[0] + "Contrastive_Learning"

# Load the filtered exon PSI file
file_name = "variable_cassette_exons"
input_path = os.path.join(base_dir, "data", "ASCOT", file_name + ".csv")
output_dir = os.path.join(base_dir, "data", "ASCOT", "psi_per_tissue", file_name)
os.makedirs(output_dir, exist_ok=True)

# Load the filtered exon PSI file
df = pd.read_csv(input_path)

# Utility: Clean tissue names to make them filename-safe
def clean_filename(tissue):
    return re.sub(r'[^a-zA-Z0-9_]', '_', tissue.strip())

# Function to parse exon_location like 'chr1:123-456'
def parse_location(loc):
    chrom, coords = loc.split(':')
    start, end = coords.split('-')
    return chrom, int(start), int(end)

# Function to calculate intron coordinates
def calculate_intron(start, end, strand):
    if strand == '+':
        if start > 200:
            return start - 200, start - 1
        else:
            return None, None
    else:
        return end + 1, end + 200

# Extract tissue columns (assumed to start at column 12 onward)
tissue_columns = df.columns[12:]

tissue_data = df[tissue_columns].replace(-1, np.nan)
df["psi_mean_across_tissues"] = tissue_data.mean(axis=1, skipna=True)


for tissue in tissue_columns:
    records = []

    for _, row in df.iterrows():
        exon_name = row["exon_id"]
        gene_id = row["gene_id"]
        strand = row["exon_strand"]

        try:
            chrom, start, end = parse_location(row["exon_location"])
        except Exception:
            continue

        intron_start, intron_end = calculate_intron(start, end, strand)
        if intron_start is None or intron_end is None:
            continue

        psi = row[tissue]
        if psi == -1 or pd.isna(psi):
            continue  # skip missing PSI values
        psi_mean = row["psi_mean_across_tissues"]
        records.append({
            "Exon Name": exon_name,
            "Species Name": "hg38",
            "Gene ID": gene_id,
            "Chromosome": chrom,
            "Exon Start": start,
            "Exon End": end,
            "Strand": strand,
            "Intron Start": intron_start,
            "Intron End": intron_end,
            "PSI": psi,
            "PSI_Mean_All_Tissues": psi_mean
        })

    # Clean and generate output file name
    safe_tissue_name = clean_filename(tissue)
    output_path = os.path.join(output_dir, f"{safe_tissue_name}_psiWmean.csv")

    pd.DataFrame(records).to_csv(output_path, index=False)
    print(f"Saved {len(records)} records to {output_path}")
