"""
Script: compare_wig.py

Description:
------------
This script is used to compare two `.wig` files and identify the differences in values at corresponding positions.
The `.wig` files are parsed to extract Chromosome, Position, and Value, and then compared to calculate the differences.
Only positions with non-zero differences are retained in the output.

Inputs:
-------
- file1: Path to the first `.wig` file (e.g., 'scores.wig').
- file2: Path to the second `.wig` file (e.g., 'scores_frankestien.wig').

Outputs:
--------
- A CSV file (`difference.csv` by default) containing positions where the values in the two files differ.
  Columns in the output file:
  - Chromosome: Chromosome name.
  - Position: Genomic position.
  - Value_file1: Value from the first `.wig` file.
  - Value_file2: Value from the second `.wig` file.
  - Difference: Difference between values (Value_file1 - Value_file2).
- Prints the DataFrame of differences to the console for inspection.

Usage Example:
--------------
1. Place the `.wig` files in the same directory as the script or specify their paths.
2. Run the script to generate a CSV file with the differences:
   python compare_wig.py
"""


import pandas as pd

def parse_wig(file_path):
    data = []
    with open(file_path, 'r') as f:
        chrom = None
        start = None
        step = 1
        for line in f:
            line = line.strip()
            if line.startswith('fixedStep'):
                # Parse fixedStep header
                tokens = line.split()
                try:
                    chrom = tokens[1].split('=')[1] if '=' in tokens[0] else None
                    start = int(tokens[2].split('=')[1])
                    step = int(tokens[3].split('=')[1])
                except IndexError:
                    print(f"Error parsing line: {line}")
                    continue  # Skip problematic lines
            elif line.startswith('track') or line == '':
                continue  # Skip metadata or empty lines
            else:
                try:
                    value = float(line)
                    data.append((chrom, start, value))
                    start += step  # Increment position for fixedStep
                except ValueError:
                    print(f"Error parsing value: {line}")
                    continue
    return pd.DataFrame(data, columns=['Chromosome', 'Position', 'Value'])

def compare_wigs(file1, file2, output_diff='difference.csv'):
    # Parse both wig files
    df1 = parse_wig(file1)
    df2 = parse_wig(file2)

    # Merge on Chromosome and Position
    merged = pd.merge(df1, df2, on=['Chromosome', 'Position'], suffixes=('_file1', '_file2'))

    # Calculate the difference
    merged['Difference'] = merged['Value_file1'] - merged['Value_file2']

    # Filter only positions where the difference is non-zero
    differences = merged[merged['Difference'] != 0]

    # Save the differences
    differences.to_csv(output_diff, index=False)

    print("Differences saved to:", output_diff)
    return differences

# Example usage:
main_dir = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/dummy/'
file1 = main_dir +'scores.wig'
file2 = main_dir +'scores_frankestien.wig'
differences = compare_wigs(file1, file2)
print(differences)
