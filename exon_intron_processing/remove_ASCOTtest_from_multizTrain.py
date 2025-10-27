import pandas as pd
import pickle
from pathlib import Path

# --- Configuration ---
# 1. Define the paths to your CSV files
csv_file_path_1 = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/test_cassette_exons_multizOverlaps.csv'  # <<< UPDATE THIS PATH
csv_file_path_2 = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/variable_cassette_exons_multizOverlaps.csv' # <<< UPDATE THIS PATH

# 2. Define the path to your input pickle file
input_pickle_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data_new/intronExonSeq_multizAlignment_noDash/trainTestVal_data/test_merged_filtered_min30Views.pkl'     # <<< UPDATE THIS PATH

# 3. Define the path for the output (filtered) pickle file
output_pickle_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data_new/intronExonSeq_multizAlignment_noDash/trainTestVal_data/test_merged_filtered_min30Views_noASCOTtstvaribl.pkl'     # <<< UPDATE THIS PATH
 # <<< UPDATE THIS PATH (optional)

# --- Function to read exon names from a CSV ---
def read_exon_names_from_csv(filepath: str) -> set:
    """Reads the 'Exon Name' column from a CSV file into a set."""
    try:
        df = pd.read_csv(filepath)
        if 'Exon Name' not in df.columns:
            raise ValueError(f"'Exon Name' column not found in {filepath}")
        return set(df['Exon Name'].unique())
    except FileNotFoundError:
        print(f"Error: CSV file not found at {filepath}")
        return set()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return set()

# --- Main Script ---
if __name__ == "__main__":
    print("Starting exon filtering process...")

    # --- Step 1: Read exon names from CSVs ---
    print(f"Reading exons to exclude from {csv_file_path_1}...")
    exons_to_exclude_1 = read_exon_names_from_csv(csv_file_path_1)
    print(f"Reading exons to exclude from {csv_file_path_2}...")
    exons_to_exclude_2 = read_exon_names_from_csv(csv_file_path_2)

    # Combine into a single set of unique exon names
    all_exons_to_exclude = exons_to_exclude_1.union(exons_to_exclude_2)

    if not all_exons_to_exclude:
        print("Warning: No exon names were found in the CSV files. No filtering will occur.")
    else:
        print(f"Found {len(all_exons_to_exclude)} unique exon names to exclude.")
        # Optional: print some examples
        # print("Examples:", list(all_exons_to_exclude)[:5])

    # --- Step 2: Load the pickle file ---
    input_path = Path(input_pickle_path)
    if not input_path.exists():
        print(f"Error: Input pickle file not found at {input_pickle_path}")
        exit()

    print(f"Loading data from {input_pickle_path}...")
    try:
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        exit()

    # Basic check if data is a dictionary (as suggested by your image)
    if not isinstance(data, dict):
        print(f"Warning: Loaded data from {input_pickle_path} is not a dictionary. Filtering might not work as expected.")
        # Attempt to proceed if it's iterable, otherwise exit
        try:
            _ = iter(data) # Check if it's iterable
        except TypeError:
             print("Error: Loaded data is not iterable (like a dict or list). Cannot filter.")
             exit()

    original_count = len(data)
    print(f"Original data contains {original_count} entries.")

    # --- Step 3: Filter the data ---
    print("Filtering data...")
    # Create a new dictionary keeping only the exons NOT in the exclusion list
    filtered_data = {
        exon_name: sequences
        for exon_name, sequences in data.items()
        if exon_name not in all_exons_to_exclude
    }
    filtered_count = len(filtered_data)
    removed_count = original_count - filtered_count
    print(f"Filtering complete. Kept {filtered_count} entries, removed {removed_count} entries.")

    # --- Step 4: Save the filtered data ---
    output_path = Path(output_pickle_path)
    print(f"Saving filtered data to {output_pickle_path}...")
    try:
        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(filtered_data, f)
        print("Filtered data saved successfully.")
    except Exception as e:
        print(f"Error saving filtered pickle file: {e}")

    print("Exon filtering process finished.")