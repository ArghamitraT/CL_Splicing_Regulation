
import pandas as pd
import numpy as np
import pickle
import glob
import os


# --- 1. Merge the Distance Matrix Pickle Files ---

print("--- Step 1: Loading and merging pickle files ---")

# Find all files matching the pattern
file_pattern = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/*_ExonExon_meanAbsDist.pkl'
pickle_files = glob.glob(file_pattern)

# Load all dataframes from the pickle files into a list
df_list = [pd.read_pickle(file) for file in pickle_files]
print(f"Found and loaded {len(df_list)} files: {pickle_files}")

# Concatenate all dataframes
# This combines them into one large dataframe.
# If there are duplicate exons, we need to resolve them.
merged_df = pd.concat(df_list)

# Handle duplicate rows/columns from the merge
# We group by the index (exon_id) and take the first entry. This ensures each exon is unique.
# We do this for both rows and columns.
merged_df = merged_df.groupby(merged_df.index).first()
merged_df = merged_df.T.groupby(merged_df.T.index).first().T

print(f"\nMerged DataFrame created with shape: {merged_df.shape}")
print("A peek at the merged DataFrame:")
print(merged_df.iloc[:4, :4])


# --- 2. Load the Exon ID CSVs ---

print("\n--- Step 2: Loading train/val/test exon IDs ---")

train_exon_ids = pd.read_csv('/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/train_exon_list.csv')['exon_id'].tolist()
val_exon_ids = pd.read_csv('/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/val_exon_list.csv')['exon_id'].tolist()
test_exon_ids = pd.read_csv('/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/test_exon_list.csv')['exon_id'].tolist()

print(f"Loaded {len(train_exon_ids)} train IDs.")
print(f"Loaded {len(val_exon_ids)} validation IDs.")
print(f"Loaded {len(test_exon_ids)} test IDs.")


# --- 3. Filter the Merged DataFrame for each Split ---

print("\n--- Step 3: Subsetting the merged matrix by exon IDs ---")

# It's good practice to find the intersection of IDs, in case some
# exons in your CSV are not in the distance matrix.
available_train_ids = list(set(train_exon_ids) & set(merged_df.index))
available_val_ids = list(set(val_exon_ids) & set(merged_df.index))
available_test_ids = list(set(test_exon_ids) & set(merged_df.index))

# Use .loc to select the rows and columns for each subset
train_dist_df = merged_df.loc[available_train_ids, available_train_ids]
val_dist_df = merged_df.loc[available_val_ids, available_val_ids]
test_dist_df = merged_df.loc[available_test_ids, available_test_ids]

print(f"\nCreated train distance matrix with shape: {train_dist_df.shape}")
print(f"Created validation distance matrix with shape: {val_dist_df.shape}")
print(f"Created test distance matrix with shape: {test_dist_df.shape}")

# Optional: Save the new DataFrames to files
train_dist_df.to_pickle('/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/train_meanAbsDist_perMultiz.pkl')
val_dist_df.to_pickle('/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/val_meanAbsDist_perMultiz.pkl')
test_dist_df.to_pickle('/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/test_meanAbsDist_perMultiz.pkl')

print("\nProcess complete.")