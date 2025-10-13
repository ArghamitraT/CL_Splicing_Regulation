import pandas as pd
import numpy as np
import glob
import os
from utils import load_csv, get_tissue_PSI_ASCOT, save_matrix, compute_meanAbsoluteDistance, compute_meanAbsoluteDistance_blockwise
import time


# --- Configuration: Set your base paths here ---
BASE_PATH = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/"
ASCOT_PATH = os.path.join(BASE_PATH, "ASCOT")
NEW_SPLIT_PATH = os.path.join(BASE_PATH, "final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data")



# --- Main Script ---

# ==============================================================================
# Step 1: Create a master mapping from all multizOverlaps files
# ==============================================================================
print("--- Step 1: Creating master ENST <-> ASCOT mapping ---")
multiz_files = glob.glob(os.path.join(ASCOT_PATH, "*_cassette_exons_multizOverlaps.csv"))
multiz_dfs = [pd.read_csv(f) for f in multiz_files]
combined_multiz_df = pd.concat(multiz_dfs).drop_duplicates(subset=['Exon Name']).reset_index(drop=True)

# Create the dictionary: Exon Name (ENST...) -> ascot_exon_id (GT...)
enst_to_ascot_map = dict(zip(combined_multiz_df['Exon Name'], combined_multiz_df['ascot_exon_id']))
# We need a reverse map for this step: ascot_exon_id (GT...) -> Exon Name (ENST...)
ascot_to_enst_map = dict(zip(combined_multiz_df['ascot_exon_id'], combined_multiz_df['Exon Name']))

print(f"Created a map with {len(enst_to_ascot_map)} unique ENST IDs.")
print(f"Master multiz overlap DataFrame shape: {combined_multiz_df.shape}")

# ==============================================================================
# Step 2: Create a mega metadata DataFrame from all cassette_exons files
# ==============================================================================
print("\n--- Step 2: Creating mega metadata DataFrame ---")
meta_files = glob.glob(os.path.join(ASCOT_PATH, "*_cassette_exons.csv"))
meta_dfs = [pd.read_csv(f) for f in meta_files]
mega_meta_df = pd.concat(meta_dfs).drop_duplicates(subset=['exon_id'])

# Set the ASCOT ID (GT...) as the index for easy lookup
mega_meta_df = mega_meta_df.set_index('exon_id')
print(f"Created mega metadata DataFrame with {len(mega_meta_df)} unique ASCOT exons.")

# ==============================================================================
# Step 3: Load new exon lists and map ENST... IDs to ASCOT GT... IDs
# ==============================================================================
print("\n--- Step 3: Loading new splits and finding corresponding ASCOT IDs ---")
new_gt_lists = {}
divisions = ["train", "val", "test"]

for division in divisions:
    list_path = os.path.join(NEW_SPLIT_PATH, f"{division}_exon_list.csv")
    print(f"Processing {list_path}...")
    new_enst_df = pd.read_csv(list_path)
    new_enst_ids = new_enst_df['exon_id'].tolist()
    
    # Map ENST... IDs to GT... IDs using the dictionary from Step 1
    # We skip any ENST IDs that might not be in our map
    new_gt_ids = [enst_to_ascot_map[enst] for enst in new_enst_ids if enst in enst_to_ascot_map]
    
    new_gt_lists[division] = new_gt_ids
    print(f"  -> Found {len(new_enst_ids)} ENST IDs, mapped to {len(new_gt_ids)} ASCOT IDs for '{division}'.")

# ==============================================================================
# Step 4: Grab the ASCOT metadata for each new split
# ==============================================================================
print("\n--- Step 4: Extracting metadata for new splits ---")
new_split_metadata_dfs = {}
for division in divisions:
    gt_ids_for_split = new_gt_lists[division]
    
    # Select rows from the mega metadata DataFrame using the list of GT... IDs
    # .reindex() is used to preserve the order and handle missing IDs gracefully
    split_meta_df = mega_meta_df.reindex(gt_ids_for_split).dropna(how='all')
    
    new_split_metadata_dfs[division] = split_meta_df
    print(f"  -> Extracted metadata for '{division}' split. Shape: {split_meta_df.shape}")



import pickle
# ==============================================================================
# Step 6: Save Processed Data for Future Use
# ==============================================================================
print("\n--- Step 6: Saving processed data objects for future use ---")

# Define an output directory to keep things organized
OUTPUT_DIR = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/dividedLikeMultiz"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Data will be saved in: {OUTPUT_DIR}")

# 1. Save the mega metadata DataFrame
mega_meta_path = os.path.join(OUTPUT_DIR, "mega_meta_df.pkl")
mega_meta_df.to_pickle(mega_meta_path)
print(f"  -> Saved mega metadata DataFrame to {mega_meta_path}")

# 2. Save the mapping dictionaries
enst_map_path = os.path.join(OUTPUT_DIR, "enst_to_ascot_map.pkl")
with open(enst_map_path, 'wb') as f:
    pickle.dump(enst_to_ascot_map, f)
print(f"  -> Saved ENST-to-ASCOT map to {enst_map_path}")

ascot_map_path = os.path.join(OUTPUT_DIR, "ascot_to_enst_map.pkl")
with open(ascot_map_path, 'wb') as f:
    pickle.dump(ascot_to_enst_map, f)
print(f"  -> Saved ASCOT-to-ENST map to {ascot_map_path}")

# 3. Save the dictionary of redistributed metadata DataFrames
split_meta_path = os.path.join(OUTPUT_DIR, "new_split_metadata_dfs.pkl")
with open(split_meta_path, 'wb') as f:
    pickle.dump(new_split_metadata_dfs, f)
print(f"  -> Saved dictionary of split metadata to {split_meta_path}")

print("\n--- All processed files have been saved. ---")