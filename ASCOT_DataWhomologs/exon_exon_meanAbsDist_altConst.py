import pandas as pd
import numpy as np
import os
from utils import load_csv, get_tissue_PSI_ASCOT, save_matrix, compute_meanAbsoluteDistance, compute_meanAbsoluteDistance_blockwise
import time
import pickle


# --- Configuration: Set your base paths here ---
BASE_PATH = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/"
ASCOT_PATH = os.path.join(BASE_PATH, "ASCOT")
NEW_SPLIT_PATH = os.path.join(BASE_PATH, "final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data")

def get_npnnanmean(expr_matrix: pd.DataFrame, psi_vector_all_ones: np.ndarray) -> float:
    """
    Computes the mean of a numpy array while ignoring NaN values.
    Returns NaN if all values are NaN.
    """
    return np.nanmean(np.abs(expr_matrix.values - psi_vector_all_ones)/100, axis=1)


def add_DScore(expr_matrix: pd.DataFrame, weight_matrix_df1: pd.DataFrame) -> None:

    # Step 1: Create the hypothetical exon's PSI vector (all 1s)
    # Its length must match the number of tissues/samples in your expr_matrix.
    n_tissues = expr_matrix.shape[1]
    psi_vector_all_ones = np.ones(n_tissues)

    # Step 2: Calculate the MAD for each exon against the "all ones" vector
    # This computes mean(|real_psi_values - 1|) for each exon.
    # mad_from_ones = np.nanmean(np.abs(expr_matrix.values - psi_vector_all_ones)/100, axis=1)
    mad_from_ones = get_npnnanmean(expr_matrix, psi_vector_all_ones)

    # Step 3: Add the result as a new column to your weight matrix
    # The new Series needs to have the same index as your weight matrix.
    weight_matrix_df1['D_score'] = pd.Series(mad_from_ones, index=expr_matrix.index)

    # --- Display the final result ---
    print("Successfully added the new column 'D_score'.")
    print(weight_matrix_df1.head())
    return weight_matrix_df1

def add_DScore_blockwise(
    expr_matrix: pd.DataFrame,
    weight_matrix_df: pd.DataFrame,
    block_size: int = 1000
) -> pd.DataFrame:
    """
    Computes and adds a 'D_score' column to the weight matrix in a memory-efficient,
    blockwise manner. Ideal for large training datasets.

    The D-score is the Mean Absolute Distance (MAD) from a hypothetical exon
    whose PSI values are all 1.

    Args:
        expr_matrix: DataFrame with expression values (N_exons × N_tissues).
        weight_matrix_df: The DataFrame to which the D_score column will be added.
        block_size: The number of exons to process in each block.

    Returns:
        The weight_matrix_df with the new 'D_score' column added.
    """
    print(f"Adding D-Score using blockwise method with block size {block_size}...")
    N, n_tissues = expr_matrix.shape

    # 1. Create the reference PSI vector (all 1s)
    psi_vector_all_ones = np.ones(n_tissues)

    # 2. Process in blocks to conserve memory
    d_scores_from_blocks = []
    for i_start in range(0, N, block_size):
        i_end = min(i_start + block_size, N)
        print(f"  Processing exons {i_start} to {i_end}...")

        # Get the current block of expression values as a numpy array
        # expr_block_values = expr_matrix.iloc[i_start:i_end].values

        # Calculate MAD for just this block
        # mad_from_ones_block = np.mean(np.abs(expr_block_values - psi_vector_all_ones), axis=1)
        mad_from_ones_block = get_npnnanmean(expr_matrix.iloc[i_start:i_end], psi_vector_all_ones)

        # Append the results of the block to our list
        d_scores_from_blocks.append(mad_from_ones_block)

    # 3. Combine the results from all blocks
    all_d_scores = np.concatenate(d_scores_from_blocks)

    # 4. Add the final, combined Series as a new column
    weight_matrix_df['D_score'] = pd.Series(all_d_scores, index=expr_matrix.index)
    print("\nSuccessfully added the new column 'D_score'.")
    print(weight_matrix_df.head())
    return weight_matrix_df


def rename_weight_matrix(weight_matrix_df: pd.DataFrame, id_to_name_map: dict) -> pd.DataFrame:
    """
    Renames the index and columns of a weight matrix DataFrame using a pre-existing dictionary.

    Args:
        weight_matrix_df (pd.DataFrame): The square weight matrix with original IDs
                                         as index and columns.
        id_to_name_map (dict): A dictionary mapping the original IDs (keys) to the
                               new names (values).

    Returns:
        pd.DataFrame: A new DataFrame with the index and columns renamed.
    """
    print(f"Using provided map with {len(id_to_name_map)} entries to rename matrix...")

    # 1. Perform the renaming operation on both the index and columns
    renamed_df = weight_matrix_df.rename(
        index=id_to_name_map,
        columns=id_to_name_map
    )

    # 2. (Optional but Recommended) Check for any IDs that were not renamed
    # This check is still useful within the function to verify its operation.
    original_ids = set(weight_matrix_df.index)
    mapped_ids = set(id_to_name_map.keys())
    unmapped_ids = original_ids - mapped_ids

    if unmapped_ids:
        print(f"\nWarning: {len(unmapped_ids)} IDs from the weight matrix were not found "
              f"in the mapping dictionary and were left unchanged.")
        # Print a few examples of unmapped IDs
        print(f"Example unmapped IDs: {list(unmapped_ids)[:5]}")
    else:
        print("Successfully renamed all IDs found in the mapping file.")

    return renamed_df   


# --- Main Script ---

# ==============================================================================
# Step 5: Loop through new splits and compute MAD matrices
# ==============================================================================
print("\n--- Step 5: Computing final MAD matrices for each new split ---")

start = time.time()

# File paths
division = "train"  # "train", "val", "test", "variable"
output_path = f"/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/all_weights/{division}_ExonExon_meanAbsDist.pkl"
SAVED_PATH = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/dividedLikeMultiz"

ascot_to_enst_map_path = os.path.join(SAVED_PATH, "ascot_to_enst_map.pkl")
with open(ascot_to_enst_map_path, "rb") as f:
    ascot_to_enst_map = pickle.load(f)

new_split_metadata_dfs_path = os.path.join(SAVED_PATH, "new_split_metadata_dfs.pkl")
with open(new_split_metadata_dfs_path, "rb") as f:
    new_split_metadata_dfs = pickle.load(f)


print(f"\nProcessing division: '{division}'")

# This is the metadata DataFrame for the current split (e.g., train)
df = new_split_metadata_dfs[division]

# Get tissue expression matrix
expr_matrix = get_tissue_PSI_ASCOT(df)

# The exon_ids from this df are the ASCOT GT... IDs (because it's the index)
ascot_ids = df.index


# Now call your function with the expression matrix and the ENST names
if division == 'train':
    mad_df = compute_meanAbsoluteDistance_blockwise(expr_matrix, ascot_ids)
    weight_matrix_df = add_DScore_blockwise(expr_matrix, mad_df)
else:   
    mad_df = compute_meanAbsoluteDistance(expr_matrix, ascot_ids)
    weight_matrix_df = add_DScore(expr_matrix, mad_df)

total_nan_count = weight_matrix_df.isnull().sum().sum()
print(f"  -> Total number of NaN values in the '{division}' matrix: {total_nan_count}")


mad_df = rename_weight_matrix(weight_matrix_df, ascot_to_enst_map)  
print(mad_df.head())
print(f"  -> Finished renaming matrix for '{division}'. Shape: {mad_df.shape}") 
# Save
save_matrix(mad_df, output_path)
print(f"✅ Correlation matrix saved to {output_path}")

end = time.time()
print(f"✅ Total runtime: {end - start:.2f} seconds")






# # --- End of added code ---

# final_mad_matrices[division] = mad_df
# print(f"  -> Finished computing MAD matrix for '{division}'. Shape: {mad_df.shape}")

# # --- Final Result ---
# print("\n--- PIPELINE COMPLETE ---")
# for division, df in final_mad_matrices.items():
#     print(f"Final shape of '{division}' MAD matrix: {df.shape}")

# # You can now access your final matrices:
# # train_mad_df = final_mad_matrices['train']
# # val_mad_df = final_mad_matrices['val']
# # test_mad_df = final_mad_matrices['test']
