"""
Script: convert_to_bed.py

Description:
This script processes exon and intron boundary information from a multiple sequence alignment (MSA) CSV file 
and generates two BED files: one for exon boundaries and another for intron boundaries. The output BED files 
are compatible with genome browsers and tools for downstream analysis.

The script handles:
1. Extracting data specific to a given species and chromosome from the MSA CSV file.
2. Calculating exon and intron boundary positions based on strand orientation.
3. Generating two separate BED files for exon and intron boundaries with standardized formats.

Inputs:
- `input_MSA_csv`: Path to the input CSV file containing MSA and exon-intron boundary data.
  - Columns include: `Species Name`, `Chromosome`, `Exon Start`, `Exon End`, `Intron Start`, `Intron End`, `Strand`, `Exon Name`.
- `species_name`: The target species (e.g., `hg38`) for which the BED files will be generated.
- `chromosome`: The chromosome (e.g., `chr19`) to filter the data.
- `output_bed_exon`: Path to the output BED file for exon boundaries.
- `output_bed_intron`: Path to the output BED file for intron boundaries.

Outputs:
1. **Exon BED File** (`output_bed_exon`): A file containing exon boundaries with four columns:
   - Chromosome
   - Start Position
   - End Position
   - Exon Name
2. **Intron BED File** (`output_bed_intron`): A file containing intron boundaries with the same format as above.

Workflow:
1. **Input Data Extraction**:
   - Reads the input CSV and filters rows for the specified species and chromosome.
2. **Boundary Calculation**:
   - Extracts and processes exon and intron boundary positions based on the strand orientation.
3. **Output Generation**:
   - Saves the exon and intron boundaries to separate BED files.

Dependencies:
- Requires Python 3.x with the `pandas` library for data manipulation.
- Uses `os` for handling file paths.

Usage:
Run the script directly with the required parameters specified in the `main()` function:
   python convert_to_bed.py

To modify the parameters for a different species or chromosome, update the `Parameters: **CHANGE (AT)**` section.

Example:
   python convert_to_bed.py

Ensure the input CSV file contains the required columns, and the specified output directories exist before running the script.

Environment:
    snakemake_env
"""



import pandas as pd
import os

# Define a function to calculate exon-intron boundaries
def calculate_boundary(start, end, strand):
    if strand == '-':
        # For positive strand
        exon_boundary_start = end
        exon_boundary_end = end + 1
        intron_boundary_start = end+1
        intron_boundary_end = end + 2
        # intron_boundary_start = start - 1
        # intron_boundary_end = start
    elif strand == '+':
        # For negative strand
        exon_boundary_start = start 
        exon_boundary_end = start +1
        # intron_boundary_start = end
        # intron_boundary_end = end + 1
        intron_boundary_start = start - 1
        intron_boundary_end = start
    else:
        return None, None, None, None
    return exon_boundary_start, exon_boundary_end, intron_boundary_start, intron_boundary_end


def conver_to_bed(input_MSA_csv, species_name, choromosome, output_bed_exon, output_bed_intron):
    # Read the CSV file
    msa_df = pd.read_csv(input_MSA_csv)

    species_msa_df = msa_df[msa_df['Species Name'] == species_name]
    species_msa_df = species_msa_df[species_msa_df['Chromosome'] == choromosome]
    
    # Initialize a list to store the rows for the BED file
    bed_records_exon = []
    bed_records_intron = []

    # Iterate through the rows and calculate exon and intron boundaries
    for idx, row in species_msa_df.iterrows():
        chromosome = row['Chromosome']
        exon_boundary_start = int(row['Exon Start'])
        exon_boundary_end = int(row['Exon End'])
        intron_boundary_start = int(row['Intron Start'])
        intron_boundary_end = int(row['Intron End'])
        strand = row['Strand']
        exon_name = row['Exon Name']
        
        # # Calculate boundaries
        # exon_boundary_start, exon_boundary_end, intron_boundary_start, intron_boundary_end = calculate_boundary(
        #     exon_start, exon_end, strand
        # )
        
        ## .bed files need to have 4 columns
        # Exon boundary
        bed_records_exon.append([chromosome, exon_boundary_start, exon_boundary_end, exon_name])
        # Intron boundary
        bed_records_intron.append([chromosome, intron_boundary_start, intron_boundary_end, exon_name])

    # Convert to DataFrame for exporting
    bed_df_exon = pd.DataFrame(bed_records_exon, columns=['Chromosome', 'Start', 'End', 'Exon_name'])
    bed_df_exon.to_csv(output_bed_exon, sep='\t', header=False, index=False)

    bed_df_intron = pd.DataFrame(bed_records_intron, columns=['Chromosome', 'Start', 'End', 'Exon_name'])
    bed_df_intron.to_csv(output_bed_intron, sep='\t', header=False, index=False)


def main():
    ######## parameters (AT) #########
    input_MSA_csv = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/knownGene.multiz100way.exonNuc_exon_intron_positions.csv'  # Replace with your file path
    species_name = 'hg38'
    choromosome = 'chr19'
    file_name = os.path.splitext(os.path.basename(input_MSA_csv))[0]
    output_bed_exon = f'/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/19_chr/bed_files/{file_name}_exon.bed'  # Output BED file name
    output_bed_intron = f'/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/19_chr/bed_files/{file_name}_intron.bed'  # Output BED file name
    ######## parameters (AT) #########

    conver_to_bed(input_MSA_csv, species_name, choromosome, output_bed_exon, output_bed_intron)

if __name__ == "__main__":
    main()