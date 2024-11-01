
import pandas as pd
import glob
import os

# Function to calculate intron boundaries based on strand orientation
def calculate_intron(start, end, strand):
    if strand == '+':
        # Check if the exon start is large enough to calculate upstream
        if start > 200:
            intron_start = start - 200
            intron_end = start - 1
        else:
            intron_start, intron_end = None, None  # Invalid upstream case
    else:
        # For the negative strand, calculate downstream (as upstream is after the exon end)
        intron_start = end + 1
        intron_end = end + 200

    return intron_start, intron_end

# Path to the MSA alignment file (replace with the actual path)
msa_file_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/alignmentknownGene.multiz100way.exonNuc_exon_intron_positions_split_file_1.csv'
gene_annotation_folder = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/gene_annotation/gene_annotation_csv'


# Load the MSA file
msa_df = pd.read_csv(msa_file_path)

# Prepare to store results
results = []

# Iterate over each CSV file in the gene annotation folder
for annotation_file in glob.glob(os.path.join(gene_annotation_folder, '*_exon_data.csv')):
    # Get the species name from the file name
    species_name = os.path.basename(annotation_file).replace('_exon_data.csv', '')
    print(f"\nProcessing species: {species_name}")

    # Load the species-specific gene annotation file
    annotation_df = pd.read_csv(annotation_file)

    # Filter MSA data for this species
    species_msa_df = msa_df[msa_df['Species Name'] == species_name]

    # Check intron-exon boundary matches
    match_count = 0
    for _, row in annotation_df.iterrows():
        exon_start = row['exon_start']
        exon_end = row['exon_end']
        strand = row['strand']
        chromosome = row['chromosome']

        # Calculate the intron boundaries
        intron_start, intron_end = calculate_intron(exon_start, exon_end, strand)

        # If valid intron boundaries are calculated, check for matches
        if intron_start is not None and intron_end is not None:
            if strand == '+':
                # For + strand, match exon_start and intron_end boundary
                match = species_msa_df[
                    (species_msa_df['Chromosome'] == chromosome) &
                    (species_msa_df['Exon Start'] == exon_start) &
                    (species_msa_df['Intron End'] == intron_end)
                ]
            else:
                # For - strand, match exon_end and intron_start boundary
                match = species_msa_df[
                    (species_msa_df['Chromosome'] == chromosome) &
                    (species_msa_df['Exon End'] == exon_end) &
                    (species_msa_df['Intron Start'] == intron_start)
                ]

            # Count matches
            if not match.empty:
                match_count += 1

    # Append results for this species
    results.append({
        "Species": species_name,
        "Total Exons": len(annotation_df),
        "Matches": match_count,
        "Match Percentage": (match_count / len(annotation_df)) * 100
    })

# Convert results to a DataFrame for easy viewing and save
results_df = pd.DataFrame(results)
results_df.to_csv('/path/to/output/intron_exon_match_results.csv', index=False)
print("Intron-exon match results saved to intron_exon_match_results.csv")
print(results_df)
