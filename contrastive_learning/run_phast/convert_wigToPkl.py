"""
Script: wig_to_pickle.py

Description:
This script processes WIG files containing genomic data and converts them into pandas DataFrame objects.
The DataFrames are saved as pickle files for faster access and downstream analysis.

The script handles:
1. Parsing WIG files with `fixedStep` format to extract positions and scores.
2. Converting the extracted data into pandas DataFrame objects.
3. Saving the processed DataFrame for each WIG file as a `.pkl` file in the specified output directory.

Inputs:
- `wig_folder`: Path to the directory containing WIG files to be processed.
  - Each WIG file is expected to use the `fixedStep` format and contain genomic positions and scores.
- `output_dir`: Path to the output directory where processed pickle files will be saved.

Outputs:
1. **Pickle Files** (`.pkl`): Each WIG file is processed and saved as a pickle file with the same name in the specified output directory.

Workflow:
1. **File Reading**:
   - Iterates over all WIG files in the specified input directory.
   - Parses genomic position (`fixedStep`) and corresponding scores.
2. **Data Conversion**:
   - Stores the extracted data as a pandas DataFrame with columns `position` and `score`.
3. **Pickle File Generation**:
   - Saves the DataFrame for each WIG file as a `.pkl` file in the output directory.

Dependencies:
- Requires Python 3.x with the `pandas` library for data manipulation.
- Uses `os` for handling file paths.

Usage:
Run the script directly with the required parameters specified in the `main()` function:
   python wig_to_pickle.py

To modify the parameters for a different input or output folder, update the `Parameters: **CHANGE (AT)**` section.

Example:
   python wig_to_pickle.py

Ensure the input directory contains valid WIG files and the specified output directory exists or is writable.

Environment:
    snakemake_env
"""
import pandas as pd
import os
import time

# Function to convert WIG to DataFrame and save as pickle
def wig_to_pickle(wig_file, output_dir):
    """
    Converts a WIG file to a pandas DataFrame and saves it as a pickle file.
    """
    data = []
    start_pos = None

    start = time.time()
    with open(wig_file, 'r') as file:
        # data = file.readlines()
        # end = time.time()
        # print(f"Elapsed time: {end - start:.2f} seconds")
        for line in file:
            # x=0
            try:
                if line.startswith('fixedStep'):
                    _, step_chrom, step_start, step_span = line.split()
                    start_pos = int(step_start.split('=')[1])
                elif float(line.strip()) and start_pos is not None:
                    data.append((start_pos, float(line.strip())))
                    start_pos += 1
            except:
                continue
        
    end = time.time()
    print(f"Elapsed time: {end - start:.2f} seconds")

    # Create DataFrame
    df = pd.DataFrame(data, columns=['position', 'score'])

    # Save as pickle
    pickle_file = os.path.join(output_dir, os.path.basename(wig_file).replace('.wig', '.pkl'))
    df.to_pickle(pickle_file)
    print(f"Saved {wig_file} as {pickle_file}")
    return pickle_file

# Function to process all WIG files
def process_all_wig_files(wig_folder, output_dir):
    """
    Processes all WIG files in a folder and converts them to pickles.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    wig_files = [os.path.join(wig_folder, f) for f in os.listdir(wig_folder) if f.endswith('.wig')]
    pickle_files = []
    for wig_file in wig_files:
        pickle_files.append(wig_to_pickle(wig_file, output_dir))
    return pickle_files

# Entry point
if __name__ == "__main__":
    ######### parameters (AT) ############
    wig_folder = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/19_chr/wig_files_original/'  # Replace with actual folder path
    output_dir = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/19_chr/wigToPkl'  # Replace with actual output folder path
    ######### parameters (AT) ############

    # Process all WIG files
    pickle_files = process_all_wig_files(wig_folder, output_dir)
    print(f"Processed {len(pickle_files)} WIG files. Pickle files saved in {output_dir}.")
