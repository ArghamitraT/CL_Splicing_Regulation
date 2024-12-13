"""
Script: wig_exon_score_analysis.py

Description:
This script processes exon regions from a BED file and calculates maximum scores across multiple WIG files. It identifies the highest score and the corresponding WIG file for each exon range by iterating through all available WIG files.

The script handles:
1. Loading exon data from a BED file.
2. Reading scores from WIG files for the specific exon ranges.
3. Iterating through multiple WIG files to identify the maximum score for each exon.
4. Saving the results, including the maximum scores and corresponding WIG files, to a CSV file.

Inputs:
- `exon_bed_file`: Path to the BED file containing exon ranges.
  - The BED file should have columns: `Chromosome`, `Start`, `End`, and `Exon_Name`.
- `wig_folder`: Directory containing WIG files to process.
  - Each WIG file is expected to be in `fixedStep` format and contain genomic positions and scores.
- `output_file`: Path to save the final results in CSV format.

Outputs:
1. **Final CSV File**: Contains the following columns:
   - `Exon_Name`: Name of the exon.
   - `Max_Score`: The maximum score for the exon across all WIG files.
   - `WIG_File`: The WIG file corresponding to the maximum score.

Workflow:
1. **File Reading**:
   - Loads the exon regions from the BED file into a pandas DataFrame.
   - Reads the WIG files and extracts scores for specific exon ranges.
2. **Iterative Processing**:
   - Iterates through exons and WIG files, calculating scores for each exon.
   - Updates the maximum score and best file name for each exon based on the calculated scores.
3. **Results Compilation**:
   - Combines results into a single DataFrame and saves the final CSV output.

Dependencies:
- Requires Python 3.x with `pandas`, `numpy`, and `os` libraries.

Usage:
Run the script with the required parameters specified in the `__main__` function:
   python wig_exon_score_analysis.py

To modify parameters for a different BED file or WIG folder, update the `Parameters: **CHANGE (AT)**` section.

Example:
   python wig_exon_score_analysis.py

Ensure the input BED file, WIG files, and required Python libraries are accessible.

Environment:
    contrastive_learning_env
"""


import pandas as pd
import os
import numpy as np

# Function to handle exon processing
def process_exons(exon_bed_file):
    """
    Reads and processes exon data from a BED file.
    """
    return pd.read_csv(exon_bed_file, sep='\t', header=None, names=['Chromosome', 'Start', 'End', 'Exon_Name'])

# Function to process WIG scores for a specific exon range
def get_scores_from_wig(wig_file, exon_start, exon_end):
    """
    Reads a WIG file and extracts scores for a specific exon range.
    """
    scores = []
    start_pos = None

    with open(wig_file, 'r') as file:
        for line in file:
            if line.startswith('fixedStep'):
                _, step_chrom, step_start, step_span = line.split()
                start_pos = int(step_start.split('=')[1])
                
            elif float(line.strip()) and start_pos is not None:
                if exon_start <= start_pos <= exon_end:
                    scores.append(float(line.strip()))
                start_pos += 1

    return np.mean(np.array(scores))


# Function to find the maximum score for each exon across all WIG files
def find_max_scores_with_wig(exons, wig_files):
    """
    Iterates over exons and processes WIG files incrementally to find maximum scores.
    """
    results = []

    for idx, exon in exons.iterrows():
        if idx >= 2:  # Limit to processing only the first 2 exons
            break
        exon_start = exon['Start']
        exon_end = exon['End']
        exon_name = exon['Exon_Name']
        print(exon_name)

        max_score = -np.inf
        best_wig_file = None

        for wig_file in wig_files:
            scores = get_scores_from_wig(wig_file, exon_start, exon_end)
            if scores:
                exon_score = scores
                if exon_score > max_score:
                    max_score = exon_score
                    best_wig_file = os.path.basename(wig_file)
            print("score ", scores)
            print(wig_file)

        results.append({'Exon_Name': exon_name, 'Max_Score': max_score, 'WIG_File': best_wig_file})

    return pd.DataFrame(results)

# Main function to orchestrate the workflow
def main(exon_bed_file, wig_folder, output_file):
    """
    Main function to process exons and WIG files, and save the results.
    """
    # Process exons
    exons = process_exons(exon_bed_file)

    # Get all WIG files
    wig_files = [os.path.join(wig_folder, f) for f in os.listdir(wig_folder) if f.endswith('.wig')]

    # Find maximum scores for exons
    result_df = find_max_scores_with_wig(exons, wig_files)

    # Save results to a CSV file
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# Entry point
if __name__ == "__main__":
    
    ######### parameters (AT) ############
    exon_bed_file = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/19_chr/bed_files/knownGene.multiz100way.exonNuc_exon_intron_positions_chr19_exon.bed'  # Replace with actual path
    wig_folder = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/19_chr/wig_files/'  # Replace with actual folder path
    output_file = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/19_chr/exon_max_scores_termnimal.csv'
    temp_dir = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/19_chr/BP_specific_scores'
    ######### parameters (AT) ############

    # Run the main functionç
    main(exon_bed_file, wig_folder, output_file)
