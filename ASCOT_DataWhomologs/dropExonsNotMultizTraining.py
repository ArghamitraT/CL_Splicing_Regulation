import pandas as pd
import pickle
import os

# --- 1. CONFIGURATION ---
# Please update these paths to match your project structure.

# Path to your CURRENT master list of exon names.
ORIGINAL_MASTER_LIST_PATH = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/all_exon_list.csv"

# Path where the NEW, updated master list will be saved.
UPDATED_MASTER_LIST_PATH = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/all_exon_list_updated.csv"

# Directory where the train/val/test alternate exon .pkl files are located.
CORRELATION_DIR = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data"

# The data splits to process.
DIVISIONS = ['train', 'val', 'test']

# --- 2. UPDATE LOGIC ---

def update_master_exon_list():
    """
    Loads the original master exon list and adds any missing alternate
    exons found in the division-specific correlation files.
    """
    print("--- Starting Master Exon List Update Process ---")

    # Load the original master list of valid exon names into a set
    try:
        master_df = pd.read_csv(ORIGINAL_MASTER_LIST_PATH)
        master_exon_set = set(master_df['exon_id'].tolist())
        original_count = len(master_exon_set)
        print(f"Loaded original master list with {original_count} unique exon names.")
    except FileNotFoundError:
        print(f"Warning: Original master list not found at: {ORIGINAL_MASTER_LIST_PATH}")
        print("Starting with an empty master list.")
        master_exon_set = set()
        original_count = 0

    # Loop through each data split to find new exons
    for division in DIVISIONS:
        print(f"--- Checking division: '{division}' ---")
        input_path = os.path.join(CORRELATION_DIR, f"all_weights/{division}_ExonExon_meanAbsDist.pkl")

        if not os.path.exists(input_path):
            print(f"File not found, skipping: {input_path}")
            continue

        with open(input_path, "rb") as f:
            weight_matrix_df = pickle.load(f)
        
        # Add the exon names from this file's index to our master set
        # The set automatically handles any duplicates
        master_exon_set.update(weight_matrix_df.index.tolist())

    # After checking all files, see how many new exons were added
    final_count = len(master_exon_set)
    new_exons_added = final_count - original_count
    print(f"\n--- Update Summary ---")
    print(f"Added {new_exons_added} new exon(s) to the master list.")
    print(f"The final master list now contains {final_count} unique exon names.")

    # Convert the final set back to a sorted list for consistent ordering
    final_exon_list = sorted(list(master_exon_set))

    # Create a new DataFrame and save it to the updated CSV file
    updated_master_df = pd.DataFrame(final_exon_list, columns=['exon_id'])
    updated_master_df.to_csv(UPDATED_MASTER_LIST_PATH, index=False)

    print(f"\nSuccessfully saved the complete master list to: {UPDATED_MASTER_LIST_PATH}")
    print("--- Process complete. ---")


if __name__ == '__main__':
    update_master_exon_list()




# import pandas as pd
# import pickle
# import os

# # --- 1. CONFIGURATION ---
# # Please update these paths to match your project structure.

# # Path to the master list of all unique, valid exon names.
# MASTER_EXON_LIST_PATH = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/all_exon_list.csv"

# # Directory where the train/val/test alternate exon .pkl files are located.
# CORRELATION_DIR = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data"

# # The data splits to process.
# DIVISIONS = ['train', 'val', 'test']

# # --- 2. PRE-PROCESSING LOGIC ---

# def clean_correlation_files():
#     """
#     Loads each division's correlation matrix, filters out exons not
#     present in the master list, and saves a new, cleaned file.
#     """
#     print("--- Starting Correlation File Cleaning Process ---")

#     # Load the master list of valid exon names into a set for fast lookups
#     try:
#         master_df = pd.read_csv(MASTER_EXON_LIST_PATH)
#         master_exon_set = set(master_df['exon_id'].tolist())
#         print(f"Successfully loaded {len(master_exon_set)} unique exon names from the master list.")
#     except FileNotFoundError:
#         print(f"FATAL ERROR: Master exon list not found at: {MASTER_EXON_LIST_PATH}")
#         return

#     # Loop through each data split (train, val, test)
#     for division in DIVISIONS:
#         print(f"\n--- Processing division: '{division}' ---")

#         input_path = os.path.join(CORRELATION_DIR, f"all_weights/{division}_ExonExon_meanAbsDist.pkl")
#         output_path = os.path.join(CORRELATION_DIR, f"all_weights_new/{division}_ExonExon_meanAbsDist.pkl")

#         if not os.path.exists(input_path):
#             print(f"Warning: Input file not found, skipping: {input_path}")
#             continue

#         # Load the correlation DataFrame
#         with open(input_path, "rb") as f:
#             weight_matrix_df = pickle.load(f)
        
#         original_count = len(weight_matrix_df)
#         print(f"Loaded '{division}' file with {original_count} alternate exons.")

#         # Identify which of these exons are valid (i.e., are in the master set)
#         current_alt_exons = weight_matrix_df.index.tolist()
#         valid_alt_exons = [name for name in current_alt_exons if name in master_exon_set]
        
#         exons_to_drop_count = original_count - len(valid_alt_exons)

#         if exons_to_drop_count > 0:
#             print(f"Found {exons_to_drop_count} exon(s) to remove (not in master list).")
            
#             # # Filter the DataFrame to keep only the rows and columns for valid exons
#             # # .loc is used to select by name. This keeps the valid rows.
#             # cleaned_df = weight_matrix_df.loc[valid_alt_exons]
#             # # Now, select the valid columns (the same names, plus the D_score column)
#             # cleaned_df = cleaned_df[valid_alt_exons + ['D_score']]
#             # --- CORRECTED LINE ---
#             # This single line robustly selects the valid rows AND columns at the same time.
#             cleaned_df = weight_matrix_df.loc[valid_alt_exons, valid_alt_exons + ['D_score']]
#             # --------------------

#             print(cleaned_df.head())
            
#             # Save the new, cleaned DataFrame
#             with open(output_path, "wb") as f:
#                 pickle.dump(cleaned_df, f)
            
#             print(f"Saved cleaned file with {len(cleaned_df)} rows and {len(cleaned_df.columns)} columns to: {output_path}")

#         else:
#             print("No inconsistencies found. All exons in this file are valid.")
#             # Optionally, you could copy the original file to the new name if you
#             # want to have consistent filenames for the next step.
#             # import shutil
#             # shutil.copy(input_path, output_path)

#     print("\n--- Cleaning process complete. ---")


# if __name__ == '__main__':
#     clean_correlation_files()