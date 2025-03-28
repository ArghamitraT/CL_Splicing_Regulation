"""
Script: compare_exon_intron_trees.py

Description:
This script compares the best trees associated with exon and intron regions to assess their matching status. 
The comparison is based on the `BigWig_File` column in two input CSV files, which indicate the best tree for 
each exon and intron, respectively.

The script handles:
1. Loading two CSV files containing exon and intron best tree data.
2. Merging the data on the `Exon_Name` column to align corresponding entries.
3. Comparing the `BigWig_File` values for exons and introns to determine if they match.
4. Calculating the percentage of matching and non-matching entries.

Inputs:
- `exon_best_tree`: Path to the CSV file containing the best trees for exons.
  - Columns: `Exon_Name`, `BigWig_File` (and other related information).
- `intron_best_tree`: Path to the CSV file containing the best trees for introns.
  - Columns: `Exon_Name`, `BigWig_File` (and other related information).

Outputs:
1. A summary printed to the console showing:
   - The percentage of matching entries.
   - The total counts of matching and non-matching rows.

Workflow:
1. **File Reading**:
   - Reads the two input CSV files into pandas DataFrames.
2. **Data Merging**:
   - Merges the exon and intron DataFrames on the `Exon_Name` column.
3. **Comparison**:
   - Compares the `BigWig_File` columns for exons and introns to identify matches.
4. **Statistics Calculation**:
   - Computes the percentage of matching rows and displays the results.

Dependencies:
- Requires Python 3.x with the `pandas` library for data manipulation.

Usage:
Run the script directly with the required parameters specified in the `main()` function:
   python compare_exon_intron_trees.py

To modify the parameters for a different input folder or files, update the `Parameters: **CHANGE (AT)**` section.

Example:
   python compare_exon_intron_trees.py

Ensure the input files exist and follow the expected format.

Environment:
    snakemake_env
"""

import pandas as pd

def top_3_frequent_values_percentage(best_tree, column_name):
    """
    Finds the top 3 most frequent values in a column and calculates the percentage of their occurrence.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Column to analyze.
        
    Returns:
        pd.DataFrame: DataFrame with the top 3 values and their percentages.
    """
    # Get value counts
    print(best_tree)
    df = pd.read_csv(best_tree, sep=',')
    value_counts = df[column_name].value_counts(normalize=True) * 100  # Normalize=True gives percentage
    
    # Get top 3 values
    top_3 = value_counts.head(10)
    
    # Print the result
    print("Top 3 most frequent values and their percentages:")
    for value, percentage in zip(top_3.index, top_3.values):
        print(f"{value}: {percentage:.2f}%")


def compare_trees(exon_best_tree, intron_best_tree):
    
    exon_best_tree_df  = pd.read_csv(exon_best_tree, sep=',')
    intron_best_tree_df  = pd.read_csv(intron_best_tree, sep=',')

    # Merge the two dataframes on Exon_Name
    merged_df = exon_best_tree_df.merge(
        intron_best_tree_df[['Exon_Name', 'Score', 'BigWig_File']],
        on='Exon_Name',
        how='outer',
        suffixes=('_exon', '_intron')
    )

    # Calculate matching status
    merged_df['matching_status'] = (merged_df['BigWig_File_exon'] == merged_df['BigWig_File_intron']).astype(int)

    # Count matching status
    counts = merged_df['matching_status'].value_counts()
    count_0 = counts.get(0, 0)  # Default to 0 if 0 is not in the column
    count_1 = counts.get(1, 0)  # Default to 0 if 1 is not in the column
    match = count_1 * 100 / (count_1 + count_0)
    print(f"match {match}%")

    # Calculate how many exons have scores greater than introns
    merged_df['exon_greater_intron'] = (merged_df['Score_exon'] > merged_df['Score_intron']).astype(int)
    exon_greater_count = merged_df['exon_greater_intron'].sum()
    print(f"Number of exons with scores greater than introns: {exon_greater_count}")




def main():
    ######### Parameters (AT) ##########
    main_dir = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/19_chr/'
    exon_best_tree = main_dir+'Best_tree_tsvFiles/knownGene.multiz100way.exonNuc_exon_intron_positions_chr19_exon_2024_12_12_21_48_39.csv'
    intron_best_tree = main_dir+'Best_tree_tsvFiles/knownGene.multiz100way.exonNuc_exon_intron_positions_chr19_intron_2024_12_12_22_00_32.csv'
    ######### Parameters (AT) ##########
    
    compare_trees(exon_best_tree, intron_best_tree)
    top_3_frequent_values_percentage(exon_best_tree, 'BigWig_File')
    top_3_frequent_values_percentage(intron_best_tree, 'BigWig_File')

if __name__=='__main__':
    main()

