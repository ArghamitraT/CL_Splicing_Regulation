"""
Script: get_maxphastcon_PKLfiles.py

Description:
This script processes exon regions from a BED file and calculates maximum scores across multiple `.pkl` files using PyRanges for efficient interval operations. It identifies the highest score and the corresponding `.pkl` file for each exon region.

The script handles:
1. Loading exon data from a BED file and `.pkl` files containing positional scores.
2. Using PyRanges for efficient overlap calculation between exon regions and positional score data.
3. Calculating the maximum score for each exon across all `.pkl` files.
4. Saving the results, including the maximum scores and corresponding `.pkl` files, to a CSV file.

Inputs:
- `exon_bed_file`: Path to the BED file containing exon ranges.
  - The BED file should have columns: `Chromosome`, `Start`, `End`, and `Exon_Name`.
- `pkl_folder`: Directory containing `.pkl` files with positional scores.
  - Each `.pkl` file is expected to have columns: `position` and `score`.
- `output_file`: Path to save the final results in CSV format.

Outputs:
1. **Final CSV File**: Contains the following columns:
   - `Chromosome`, `Start`, `End`, `Exon_Name`: Exon details.
   - `Max_Score`: The maximum score for the exon across all `.pkl` files.
   - `Best_File`: The `.pkl` file corresponding to the maximum score.

Workflow:
1. **File Reading**:
   - Loads the exon regions from the BED file into a pandas DataFrame.
   - Converts the exon DataFrame into a PyRanges object for efficient overlap operations.
2. **Batch Processing**:
   - Iterates through all `.pkl` files in the specified folder.
   - For each `.pkl` file, calculates overlaps with exon regions and computes mean scores.
3. **Results Compilation**:
   - Updates the maximum score and best file name for each exon based on the computed scores.
4. **Output Generation**:
   - Saves the results to a CSV file for downstream analysis.

Dependencies:
- Requires Python 3.x with `pandas`, `numpy`, `os`, `time`, and `datetime` libraries.
- Requires `pyranges` for efficient interval operations.

Usage:
Run the script directly with the required parameters specified in the `__main__` function:
   python exon_pyranges_analysis.py

To modify parameters for a different BED file or `.pkl` folder, update the `Parameters: **CHANGE (AT)**` section.

Example:
   python exon_pyranges_analysis.py

Ensure the input BED file, `.pkl` files, and required Python libraries are accessible.

Environment:
    contrastive_learning_env
"""

import os
import pandas as pd
import numpy as np
import time
import datetime
import pyranges as pr

def process_exons_with_pyranges(exon_bed_file, pkl_folder, output_file):
    """
    Processes exons with scores from .pkl files using PyRanges for efficient interval overlaps.

    Args:
        exon_bed_file (str): Path to the BED file containing exon ranges.
        pkl_folder (str): Path to the folder containing .pkl files.
        output_file (str): Path to save the results.
    """
    # Read exon ranges into a DataFrame
    exons_df = pd.read_csv(
        exon_bed_file,
        sep="\t",
        header=None,
        names=["Chromosome", "Start", "End", "Exon_Name"]
    )
    # Ensure Chromosome names are consistent
    exons_df['Chromosome'] = exons_df['Chromosome'].astype(str)

    # Convert exons DataFrame to PyRanges object
    exons_pr = pr.PyRanges(exons_df)

    # Initialize Max_Score and Best_File columns
    exons_df["Max_Score"] = -np.inf
    exons_df["Best_File"] = None

    # Get all .pkl files
    pkl_files = [os.path.join(pkl_folder, f) for f in os.listdir(pkl_folder) if f.endswith('.pkl')]

    start = time.time()
    # Process each .pkl file
    for pkl_file in pkl_files:
        print(f"Processing file: {os.path.basename(pkl_file)}")
        wig_data = pd.read_pickle(pkl_file)

        # Prepare wig_data for PyRanges
        wig_data_df = pd.DataFrame({
            'Chromosome': 'chr19',  # Adjust if necessary
            'Start': wig_data['position'],
            'End': wig_data['position'] + 1,  # Positions are single-base
            'Score': wig_data['score']
        })

        # Convert wig_data to PyRanges object
        wig_pr = pr.PyRanges(wig_data_df)

        # Join exons with wig_data
        overlaps = exons_pr.join(wig_pr, suffix='_wig')

        # Compute mean scores for each exon
        if overlaps.df.empty:
            continue  # No overlaps in this file
        
        end = time.time()
        print(f"time taked {end-start}s")
        mean_scores = overlaps.df.groupby(['Chromosome', 'Start', 'End', 'Exon_Name'], as_index=False)['Score'].mean()
        mean_scores.rename(columns={'Score': 'Mean_Score'}, inplace=True)

        # Merge mean_scores with exons_df on exon coordinates
        exons_df = exons_df.merge(mean_scores, on=['Chromosome', 'Start', 'End', 'Exon_Name'], how='left')

        # Update Max_Score and Best_File
        higher_scores = exons_df['Mean_Score'] > exons_df['Max_Score']
        exons_df.loc[higher_scores, 'Max_Score'] = exons_df.loc[higher_scores, 'Mean_Score']
        exons_df.loc[higher_scores, 'Best_File'] = os.path.basename(pkl_file)

        # Drop the 'Mean_Score' column for the next iteration
        exons_df.drop(columns=['Mean_Score'], inplace=True)

    # Save the final results
    exons_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
# Example usage
if __name__ == "__main__":
    crnt_tm = datetime.datetime.now()
    
    ######### parameters (AT) ############
    exon_bed_file = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/19_chr/bed_files/knownGene.multiz100way.exonNuc_exon_intron_positions_chr19_exon.bed'
    pkl_folder = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/19_chr/wigToPkl/'
    output_file = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/19_chr/'
    name = output_file+ 'exon_max_scores_terminal.csv'
    ######### parameters (AT) ############
    final_output_file = (name+"_" + str(crnt_tm.year) + "_" + str(crnt_tm.month) + "_" + str(crnt_tm.day) + "_"
                  + time.strftime("%H_%M_%S") + 'csv')
    process_exons_with_pyranges(exon_bed_file, pkl_folder, final_output_file)
