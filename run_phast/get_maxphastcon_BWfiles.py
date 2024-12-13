"""
Script: get_maxphastcon_BWfiles.py

Description:
This script calculates the maximum exon scores across multiple BigWig files using the `bigWigAverageOverBed` tool. 
It processes a BED file containing exon regions and identifies the highest conservation score and the corresponding 
BigWig file for each exon.

The script handles:
1. Iterating through all BigWig files in a specified directory.
2. Running `bigWigAverageOverBed` in batch mode to compute scores for all exons in the BED file.
3. Merging scores across BigWig files to track the maximum score and corresponding file for each exon.
4. Saving the final results to a CSV file for downstream analysis.

Inputs:
- `exon_bed_file`: Path to the BED file containing exon regions to be processed.
  - The BED file should contain exon names and coordinates.
- `bigwig_folder`: Directory containing BigWig files to be processed.
  - Each file will be analyzed to calculate conservation scores for the exons.
- `temp_dir`: Directory for storing intermediate TSV outputs from `bigWigAverageOverBed`.
- `output_file`: Path to save the final CSV file containing the highest scores and corresponding BigWig files.

Outputs:
1. **Intermediate TSV Files**: One TSV file for each BigWig file, storing scores for all exons.
2. **Final CSV File**: A CSV file containing:
   - `Exon_Name`: Name of the exon.
   - `Score`: Maximum score observed for the exon.
   - `BigWig_File`: The BigWig file with the maximum score.

Workflow:
1. **Setup**:
   - Ensures the temporary and output directories exist.
2. **Batch Processing**:
   - Executes `bigWigAverageOverBed` for each BigWig file.
   - Parses the resulting TSV files and compares scores to identify the maximum for each exon.
3. **Results Compilation**:
   - Combines results into a single DataFrame and saves the final CSV output.

Dependencies:
- Requires `bigWigAverageOverBed` to be installed and available in the environment.
- Python 3.x with the `pandas`, `os`, `numpy`, `subprocess`, and `datetime` libraries.

Usage:
Run the script with the required parameters specified in the `main()` function:
   python bigwig_exon_score_analysis.py

To modify parameters for a different BED file or BigWig directory, update the `Parameters: **CHANGE (AT)**` section.

Example:
   python bigwig_exon_score_analysis.py

Ensure the BigWig files, BED file, and required tools are accessible in the specified paths.

Environment:
    phastcon_env
"""


import pandas as pd
import os
import numpy as np
import subprocess
import datetime
import time
import matplotlib.pyplot as plt

# Function to run bigWigAverageOverBed for all exons at once
def run_bigWigAverageOverBed_batch(input_bw, input_bed, output_tsv):
    """
    Runs the bigWigAverageOverBed command for all regions specified in the BED file.
    """
    command = f"bigWigAverageOverBed {input_bw} {input_bed} {output_tsv}"
    subprocess.run(command, shell=True, check=True)
    return output_tsv

def plot_distributions_def(all_scores_df):
    """
    Plots score distributions for each exon, highlighting only the file name of the maximum value on the x-axis.
    """
    exons = all_scores_df['Exon_Name'].unique()
    fig, axes = plt.subplots(len(exons), 1, figsize=(10, 5 * len(exons)))

    if len(exons) == 1:
        axes = [axes]  # Ensure axes is iterable for a single exon

    for ax, exon in zip(axes, exons):
        exon_data = all_scores_df[all_scores_df['Exon_Name'] == exon]
        
        # Plot the bar chart
        ax.bar(range(len(exon_data)), exon_data['mean'])
        
        # Get the index of the max score
        max_idx = exon_data['mean'].idxmax()
        max_score = exon_data.loc[max_idx, 'mean']
        max_file = exon_data.loc[max_idx, 'BigWig_File']

        # Highlight the max value on the x-axis
        ax.set_xticks([max_idx])
        ax.set_xticklabels([max_file], rotation=45, fontsize=8)
        
        # Set titles and labels
        ax.set_title(f"Score Distribution for {exon}")
        ax.set_xlabel("BigWig File (Max File Highlighted)")
        ax.set_ylabel("Mean Score")
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_file = os.path.join(temp_dir, "3Intron.png")
    plt.savefig(plot_file)
    print(f"Score distribution plots saved to {plot_file}")
    plt.show()



# Function to find maximum scores across all BigWig files
def find_max_scores_batch(exon_bed_file, bigwig_files, temp_dir, save_all_scores=True, plot_distributions=True):
    """
    Uses bigWigAverageOverBed to process the entire BED file at once for all BigWig files.
    """
    results = []

    # Ensure the temporary directory exists
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Initialize a DataFrame to store the best scores
    best_scores_df = pd.DataFrame(columns=['Exon_Name', 'Score', 'BigWig_File'])
    # DataFrame to store all scores if save_all_scores is True
    all_scores_df = pd.DataFrame()


    idx = 0
    start = time.time()
    for bigwig_file in bigwig_files:
        print(idx, bigwig_file)
        idx+=1
        output_tsv = os.path.join(temp_dir, f"{os.path.basename(bigwig_file)}.tsv")
        run_bigWigAverageOverBed_batch(bigwig_file, exon_bed_file, output_tsv)

        # Read scores from the output TSV
        scores_df = pd.read_csv(output_tsv, sep='\t', header=None, names=['Exon_Name', 'Size', 'Covered', 'sum', 'mean0', 'mean'])

        # Add the current file name as a column
        scores_df['BigWig_File'] = os.path.basename(bigwig_file)
        if save_all_scores:
            all_scores_df = pd.concat([all_scores_df, scores_df[['Exon_Name', 'mean', 'BigWig_File']]], ignore_index=True)

        if best_scores_df.empty:
            # If the best_scores_df is empty, initialize it with scores from the first file
            best_scores_df = scores_df[['Exon_Name', 'mean', 'BigWig_File']].rename(columns={'mean': 'Score'})
        else:
            # Merge the current scores with the best scores DataFrame
            merged_df = best_scores_df.merge(
                scores_df[['Exon_Name', 'mean', 'BigWig_File']],
                on='Exon_Name',
                how='outer',
                suffixes=('_Best', '_Current')
            )

            # Update the best scores based on comparisons
            merged_df['Best_Score'] = merged_df[['Score', 'mean']].max(axis=1)
            merged_df['Best_File'] = merged_df.apply(
                lambda row: row['BigWig_File_Current'] if row['mean'] > row['Score'] else row['BigWig_File_Best'],
                axis=1
            )

            # Keep only relevant columns for the next iteration
            best_scores_df = merged_df[['Exon_Name', 'Best_Score', 'Best_File']].rename(columns={'Best_File': 'BigWig_File', 'Best_Score': 'Score'})
            # best_scores_df = best_scores_df[['Exon_Name', 'Best_Score', 'Best_File']].rename(columns={'Best_File': 'BigWig_File'})
    end = time.time()
    print(f"time to run {(end-start)/60} min")

    if plot_distributions:
        plot_distributions_def(all_scores_df)
        
    # Save the final best scores DataFrame
    return best_scores_df

# Main function to orchestrate the workflow
def main(exon_bed_file, bigwig_folder, output_file, temp_dir):
    """
    Main function to process exons and BigWig files, and save the results.
    """
    # Get all BigWig files
    bigwig_files = [os.path.join(bigwig_folder, f) for f in os.listdir(bigwig_folder) if f.endswith('.bw')]

    # Find maximum scores using batch processing
    result_df = find_max_scores_batch(exon_bed_file, bigwig_files, temp_dir)

    # Save results to a CSV file
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# Entry point
if __name__ == "__main__":
    
    ######### parameters (AT) ############
    main_dir = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/19_chr/'
    # exon_bed_file = main_dir+'bed_files/knownGene.multiz100way.exonNuc_exon_intron_positions_chr19_intron.bed'  # Replace with actual path
    # exon_bed_file = main_dir+'bed_files/knownGene.multiz100way.exonNuc_exon_intron_positions_chr19_exon.bed'  # Replace with actual path
    exon_bed_file = main_dir+'bed_files/3Intron.bed'  # Replace with actual path
    bigwig_folder = main_dir+'bw_files/'  # Replace with actual folder path
    output_dir = main_dir+'Best_tree_tsvFiles/'
    temp_dir = main_dir+'BP_specific_scores/'
    ######### parameters (AT) ############

    # Run the main function
    # Extract the base name (dummy.bed)
    basename = os.path.basename(exon_bed_file)
    filename_without_extension = os.path.splitext(basename)[0]
    name = os.path.join(output_dir, filename_without_extension)
    crnt_tm = datetime.datetime.now()
    output_file = (name+"_" + str(crnt_tm.year) + "_" + str(crnt_tm.month) + "_" + str(crnt_tm.day) + "_"
                  + time.strftime("%H_%M_%S")+'.csv')
    main(exon_bed_file, bigwig_folder, output_file, temp_dir)



