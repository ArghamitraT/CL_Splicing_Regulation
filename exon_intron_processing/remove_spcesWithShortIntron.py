import pickle
from pathlib import Path

# --- Configuration ---
# 1. Define the path to your input training pickle file
input_pickle_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/train_merged_filtered_min30Views.pkl'     # <<< UPDATE THIS PATH
    # <<< UPDATE THIS PATH

# 2. Define the path for the output (filtered) pickle file
output_pickle_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/train_merged_filtered_min30Views_len_filtered.pkl' # <<< UPDATE THIS PATH (Suggest adding '_len_filtered')
# 3. Define the maximum allowed length (inclusive)
MAX_INTRON_LEN = 301

# --- Main Script ---
if __name__ == "__main__":
    print("Starting intron length filtering process...")

    input_path = Path(input_pickle_path)
    if not input_path.exists():
        print(f"Error: Input pickle file not found at {input_pickle_path}")
        exit()

    # --- Step 1: Load the pickle file ---
    print(f"Loading data from {input_pickle_path}...")
    try:
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        exit()

    if not isinstance(data, dict):
        print(f"Error: Loaded data from {input_pickle_path} is not a dictionary.")
        exit()

    original_exon_count = len(data)
    print(f"Original data contains {original_exon_count} exons.")

    # --- Step 2: Filter the data ---
    print(f"Filtering species with intron lengths < {MAX_INTRON_LEN}...")
    filtered_data = {}
    species_removed_count = 0
    exons_emptied_count = 0

    # Iterate through each exon
    for exon_name, species_dict in data.items():
        if not isinstance(species_dict, dict):
            print(f"Warning: Skipping exon '{exon_name}' as its value is not a dictionary.")
            continue

        filtered_species_for_exon = {}
        # Iterate through species for the current exon
        for species, sequences in species_dict.items():
            try:
                intron_5p = sequences.get('5p', '') # Use .get for safety
                intron_3p = sequences.get('3p', '') # Use .get for safety

                # Check lengths
                if len(intron_5p) >= MAX_INTRON_LEN and len(intron_3p) >= MAX_INTRON_LEN:
                    # Keep this species if both introns are within the limit
                    filtered_species_for_exon[species] = sequences
                else:
                    # Increment counter if species is removed
                    species_removed_count += 1
                    # Optional: Print which species/exon is removed
                    # print(f"  Removing species '{species}' from exon '{exon_name}' (Lengths: 5p={len(intron_5p)}, 3p={len(intron_3p)})")

            except Exception as e:
                print(f"Warning: Error processing species '{species}' for exon '{exon_name}': {e}. Skipping this species.")
                continue

        # Add the exon to the filtered data only if it still has valid species
        if filtered_species_for_exon:
            filtered_data[exon_name] = filtered_species_for_exon
        elif exon_name in data: # Check if the exon existed originally
             exons_emptied_count += 1
             # Optional: Print which exons became empty
             # print(f"  Exon '{exon_name}' became empty after filtering.")


    filtered_exon_count = len(filtered_data)
    print(f"Filtering complete.")
    print(f"  - Species removed due to length: {species_removed_count}")
    print(f"  - Exons remaining: {filtered_exon_count} (out of {original_exon_count})")
    print(f"  - Exons that became empty (and were removed): {exons_emptied_count}")


    # --- Step 3: Save the filtered data ---
    output_path = Path(output_pickle_path)
    print(f"Saving filtered data to {output_pickle_path}...")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(filtered_data, f)
        print("Filtered data saved successfully.")
    except Exception as e:
        print(f"Error saving filtered pickle file: {e}")

    print("Intron length filtering process finished.")