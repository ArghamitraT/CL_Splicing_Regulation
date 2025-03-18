"""
Given the MSA it finds how many alignments falls into exon intron boundary of that speceies. 
this file also works with other gene annotation files
for ENSEMBL annotations
"""

import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import time
timestamp = time.strftime("%Y%m%d_%H%M%S")


# Function to calculate intron boundaries based on strand orientation
def calculate_intron_vectorized(annotation_df):
    intron_starts = []
    intron_ends = []
    
    for _, row in annotation_df.iterrows():
        start = row['exon_start']
        end = row['exon_end']
        strand = row['strand']
        
        if strand == '+':
            if start > 200:
                intron_starts.append(start - 200)
                intron_ends.append(start - 1)
            else:
                intron_starts.append(None)
                intron_ends.append(None)
        else:
            intron_starts.append(end + 1)
            intron_ends.append(end + 200)

    annotation_df['intron_start'] = intron_starts
    annotation_df['intron_end'] = intron_ends
    return annotation_df

# Path to the MSA alignment file (replace with the actual path)
msa_file_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/knownGene.multiz100way.exonNuc_exon_intron_positions.csv'
# msa_file_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/Brac1.csv'
# msa_file_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/ncbiRefSeq.multiz100way.exonNuc_exon_intron_positions.csv'

common_names_df = pd.read_csv('/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/species_refSeq_urls2.csv', header=None, names=['Species_Code', 'URL', 'Common_Name'])
gene_annotation_folder = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/Ensembl_geneAnnotaion/homo_sapiens/'
# gene_annotation_folder = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/gene_annotation/dummy'

# Load the MSA file
msa_df = pd.read_csv(msa_file_path)

# Prepare to store results
results = []
# species_arr = ['hg38', 'panTro4', 'mm10', 'canFam3', 'monDom5', 'taeGut2', 'allMis1', 'fr3']
# species_arr = ['panTro4','fr3']
species_arr = ['ncbiRefSeq']    #(AT)
species_path_arr =[]
for species in species_arr:
    species_path_arr.append(os.path.join(gene_annotation_folder, f'{species}_exon_data.csv'))


# Iterate over each CSV file in the gene annotation folder
 #(AT)
for annotation_file in glob.glob(os.path.join(gene_annotation_folder, '*_exon_data.csv')):
# for annotation_file in species_path_arr:
    # Get the species name from the file name
    species_name = os.path.basename(annotation_file).replace('_exon_data.csv', '')
    species_name = 'hg38' ##(AT)
    print(f"\nProcessing species: {species_name}")

    # Load the species-specific gene annotation file
    annotation_df = pd.read_csv(annotation_file)
    annotation_df['exon_start'] = annotation_df['exon_start'] + int(1)

    """
    # Filter MSA data for this species
    species_msa_df = msa_df[msa_df['Species Name'] == species_name]

    # Calculate intron boundaries
    # annotation_df = calculate_intron_vectorized(annotation_df)

    # For + strand, match exon_start and intron_end boundary
    ##(AT)
    # plus_matches = pd.merge(
    #     annotation_df[annotation_df['strand'] == '+'],
    #     species_msa_df,
    #     left_on=['chromosome', 'exon_start', 'intron_end'],
    #     right_on=['Chromosome', 'Exon Start', 'Intron End']
    # ).drop_duplicates(subset=['Chromosome', 'Exon Start', 'Intron End'])

    # # For - strand, match exon_end and intron_start boundary
    # minus_matches = pd.merge(
    #     annotation_df[annotation_df['strand'] == '-'],
    #     species_msa_df,
    #     left_on=['chromosome', 'exon_end', 'intron_start'],
    #     right_on=['Chromosome', 'Exon End', 'Intron Start']
    # ).drop_duplicates(subset=['Chromosome', 'Exon End', 'Intron Start'])
    # match_count = len(plus_matches) + len(minus_matches)

    # plus_matches = pd.merge(
    #     annotation_df,
    #     species_msa_df,
    #     left_on=['chromosome', 'exon_start'],
    #     right_on=['Chromosome', 'Exon Start']
    # ).drop_duplicates(subset=['Chromosome', 'Exon Start'])
    # match_count = len(plus_matches)

    ## For + strand, match exon_start and intron_end boundary
    #(AT)
    plus_matches = pd.merge(
        annotation_df[annotation_df['strand'] == '+'],
        species_msa_df,
        left_on=['chromosome', 'exon_start'],
        right_on=['Chromosome', 'Exon Start']
    ).drop_duplicates(subset=['Chromosome', 'Exon Start'])

    # For - strand, match exon_end and intron_start boundary
    minus_matches = pd.merge(
        annotation_df[annotation_df['strand'] == '-'],
        species_msa_df,
        left_on=['chromosome', 'exon_end'],
        right_on=['Chromosome', 'Exon End']
    ).drop_duplicates(subset=['Chromosome', 'Exon End'])
    match_count = len(plus_matches) + len(minus_matches)

    
    # Total matches
    
    print(f"Match Percentage: {(match_count / len(species_msa_df)) * 100}")
    """

    # Filter species-specific DataFrame
    species_msa_df = msa_df[msa_df['Species Name'] == species_name]

    # Get unique exon start and end positions from species_msa_df
    # unique_exon_starts = species_msa_df[['Chromosome', 'Exon Start']].drop_duplicates()
    # unique_exon_ends = species_msa_df[['Chromosome', 'Exon End']].drop_duplicates()
    
    # Step 1: Divide species_msa_df into positive and negative strands
    positive_strand_df = species_msa_df[species_msa_df['Strand'] == '+']
    negative_strand_df = species_msa_df[species_msa_df['Strand'] == '-']
    unique_exon_starts = positive_strand_df[['Chromosome', 'Exon Start']].drop_duplicates()      # Step 2: For positive strand, take unique exon starts
    unique_exon_ends = negative_strand_df[['Chromosome', 'Exon End']].drop_duplicates()         # Step 3: For negative strand, take unique exon ends


    # For + strand, match exon_start boundary
    plus_matches = pd.merge(
        annotation_df[annotation_df['strand'] == '+'],
        unique_exon_starts,
        left_on=['chromosome', 'exon_start'],
        right_on=['Chromosome', 'Exon Start']
    ).drop_duplicates(subset=['Chromosome', 'Exon Start'])

    # For - strand, match exon_end boundary
    minus_matches = pd.merge(
        annotation_df[annotation_df['strand'] == '-'],
        unique_exon_ends,
        left_on=['chromosome', 'exon_end'],
        right_on=['Chromosome', 'Exon End']
    ).drop_duplicates(subset=['Chromosome', 'Exon End'])

    # # Initialize empty lists to store matches
    # plus_matches = []
    # minus_matches = []

    # # Loop over each row in annotation_df
    # for _, row in annotation_df.iterrows():
    #     # Check if the row is on the positive strand
    #     if row['strand'] == '+':
    #         # Find matching rows in unique_exon_starts based on Chromosome and Exon Start
    #         match = unique_exon_starts[(unique_exon_starts['Chromosome'] == row['chromosome']) & 
    #                                 (unique_exon_starts['Exon Start'] == row['exon_start'])]
    #         # Append matching rows to plus_matches if any found
    #         if not match.empty:
    #             plus_matches.append(row)

    #     # Check if the row is on the negative strand
    #     elif row['strand'] == '-':
    #         # Find matching rows in unique_exon_ends based on Chromosome and Exon End
    #         match = unique_exon_ends[(unique_exon_ends['Chromosome'] == row['chromosome']) & 
    #                                 (unique_exon_ends['Exon End'] == row['exon_end'])]
    #         # Append matching rows to minus_matches if any found
    #         if not match.empty:
    #             minus_matches.append(row)

    # # Convert the lists to DataFrames
    # plus_matches = pd.DataFrame(plus_matches)
    # minus_matches = pd.DataFrame(minus_matches)

    # # Drop duplicates based on Chromosome and Exon Start/End
    # plus_matches = plus_matches.drop_duplicates(subset=['chromosome', 'exon_start'])
    # minus_matches = minus_matches.drop_duplicates(subset=['chromosome', 'exon_end'])


    # Calculate match count and match percentage based on unique boundaries
    match_count = len(plus_matches) + len(minus_matches)
    total_unique_boundaries = len(unique_exon_starts) + len(unique_exon_ends)

    # Match percentage
    match_percentage = (match_count / total_unique_boundaries) * 100
    print(f"Match Percentage: {match_percentage}")
    

    # Append results for this species
    results.append({
        "Species": species_name,
        "Total Exons": len(species_msa_df),
        "Matches": match_count,
        "Match Percentage": match_percentage
    })

# Convert results to a DataFrame for easy viewing and save
results_df = pd.DataFrame(results)
# Calculate non-matches for each species
results_df['Non-matches'] = results_df['Total Exons'] - results_df['Matches']
# Merge `results_df` with `common_names_df` on the species code to get common names
results_df = results_df.merge(common_names_df[['Species_Code', 'Common_Name']], left_on='Species', right_on='Species_Code', how='left')
# Save the results DataFrame
results_csv_path = f'/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/intron_exon_match_results_EXONSTART{timestamp}.csv'
# results_df.to_csv(results_csv_path, index=False) #(AT)
# print(f"Intron-exon match results saved to {results_csv_path}")

# Exclude hg38 for plotting and calculate the average match percentage
# filtered_results_df = results_df[results_df['Species'] != 'hg38']
filtered_results_df = results_df #(AT)
average_match_percentage = filtered_results_df['Match Percentage'].mean()

# Add the "Average" row to the DataFrame for plotting
average_row = pd.DataFrame({
    'Species': ['Average'],
    'Total Exons': [filtered_results_df['Total Exons'].mean()],
    'Matches': [filtered_results_df['Matches'].mean()],
    'Match Percentage': [average_match_percentage],
    'Common_Name': ['Average']  # Label for the average bar
})
plot_df = pd.concat([filtered_results_df, average_row], ignore_index=True)

# Create a combined label for x-axis with both common name and scientific code
plot_df['Label'] = plot_df.apply(lambda row: f"{row['Common_Name']} ({row['Species']})" if row['Species'] != 'Average' else 'Average', axis=1)

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot only the match percentage as bars
bars = ax.bar(plot_df['Label'], plot_df['Match Percentage'], label='Matches (%)', color='skyblue')

# Annotate each bar with the match percentage and total matches
for bar, match_pct, matches in zip(bars, plot_df['Match Percentage'], plot_df['Matches']):
    # Get the height of the bar to place the annotation
    height = bar.get_height()
    ax.annotate(
        # f"{match_pct:.2f}%\n({int(matches)})",  # Show percentage and total matches
        f"{match_pct:.2f}",  # Show percentage and total matches
        # f"({int(matches)})",  # Show percentage and total matches
        xy=(bar.get_x() + bar.get_width() / 2, height),  # Position at the center-top of each bar
        xytext=(0, 8),  # Offset text by 8 points vertically
        textcoords="offset points",
        ha='center', va='bottom'
    )

# Add a horizontal line for the average match percentage (excluding hg38)
ax.axhline(y=average_match_percentage, color='green', linestyle='--', label=f'Average Match % (Excl. hg38): {average_match_percentage:.2f}%')

# Customize the plot
ax.set_xlabel('Species')
ax.set_ylabel('Match Percentage (%)')
ax.set_title('Exon Match Percentage (Human, ensembl)') #(AT)
ax.legend()

# Rotate x-axis labels for better readability
plt.xticks(rotation=90)

# Save and show the plot with a timestamped filename
timestamp = time.strftime("%Y%m%d_%H%M%S")
plot_file_path = f'/gpfs/commons/home/atalukder/Contrastive_Learning/code/figures/intron_exon_match_results_{timestamp}.png'
plt.tight_layout()
plt.savefig(plot_file_path)
print(f"Plot saved to {plot_file_path}")
plt.show()

