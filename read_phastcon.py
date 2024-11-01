import os
import subprocess

def run_bigWigAverageOverBed(input_bw, input_dir, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get a list of all files in the input directory
    files = os.listdir(input_dir)
    
    for file in files:
        if file.endswith('.bed'):  # Only process BED files
            # Construct the output file name based on the input file name
            output_tsv = os.path.join(output_dir, f"{file}.tsv")
            input_bed = os.path.join(input_dir, file)

            # Construct and run the command
            command = f"bigWigAverageOverBed {input_bw} {input_bed} {output_tsv}"
            subprocess.run(command, shell=True, check=True)
            print(f"Processed {file}")

# Example usage
input_bw = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/hg38.phastCons100way.bw"  # Path to the .bw file
input_dir = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/"  # Directory containing BED files
output_dir = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/"  # Directory to save TSV output files

run_bigWigAverageOverBed(input_bw, input_dir, output_dir)



# import pandas as pd

# # Load the CSV file
# input_csv = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/dummy_exon_intron_positions_shrt.csv'  # Replace with your file path
# output_bed = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/dummy_exon_intron_positions_shrt.bed'  # Output BED file name

# # Read the CSV file
# df = pd.read_csv(input_csv)

# # Select the required columns and add a placeholder column
# #  Columns required are: chromosome, start, end, name (4 fields)
# bed_df = df[['Chromosome', 'Exon Start', 'Exon End']]

# # Ensure Exon Start and Exon End are integers
# bed_df['Exon Start'] = bed_df['Exon Start'].astype(int)
# bed_df['Exon End'] = bed_df['Exon End'].astype(int)

# # Remove duplicates again to be thorough
# bed_df = bed_df.drop_duplicates(subset=['Chromosome', 'Exon Start', 'Exon End'])

# # Add a unique identifier as the fourth column to avoid duplication issues
# bed_df['UniqueID'] = range(1, len(bed_df) + 1)

# # Save as a BED file with tab-separated values
# bed_df.to_csv(output_bed, sep='\t', header=False, index=False)

# print(f"BED file saved as {output_bed}")