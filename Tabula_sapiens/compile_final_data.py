import pandas as pd
import numpy as np
from functools import reduce
import os
import json
import argparse
from scipy.special import logit

def main():
    parser = argparse.ArgumentParser(description="Combine intermediate tables to final data")
    parser.add_argument("--main_dir", required=True, help="File path to Tabula Sapiens directory")  # "/gpfs/commons/home/nkeung/tabula_sapiens"
    args = parser.parse_args()

    main_dir = args.main_dir
    
    with open(os.path.join(main_dir, "completed.json"), "r") as f:
        all_cell_types = list(json.load(f))
    # all_cell_types = ["pericyte", "mesenchymal_stem_cell_of_adipose_tissue", "ltf+_epithelial_cell"]    # Test set chosen at random
    all_cells_set = set([name.replace("_", " ") for name in all_cell_types])

    csv_dir = os.path.join(main_dir, "psi_data")

    # Columns to merge on
    merge_keys = ["gene_id", "exon_location", "exon_strand", "chromosome"]

    master_df = pd.DataFrame()  # Initialize empty dataframe
    original_order = None       # Save column order

    processed = 0
    # reduce should be able to speed things up now
    # Main source of conflict came from exon_boundary. Removing it should allow reduce to work
    for i, cell_type in enumerate(all_cell_types):
        csv_file = os.path.join(csv_dir, f"{cell_type}.csv")
        df = pd.read_csv(csv_file)
        if i == 0:
            original_order = df.columns.tolist()

        # Name of PSI column for this cell type
        psi_col = cell_type.replace("_", " ")

        if master_df.empty:
            # First cell type: initialize master_df
            master_df = df.copy()
        else:
            # Set merge keys as index for upsert alignment
            master_indexed = master_df.set_index(merge_keys)
            df_indexed = df.set_index(merge_keys)

            # Existing rows
            existing_idx = master_indexed.index.intersection(df_indexed.index)
            master_indexed.loc[existing_idx, psi_col] = df_indexed.loc[existing_idx, psi_col]

            # New rows
            new_idx = df_indexed.index.difference(master_indexed.index)
            new_rows = df_indexed.loc[new_idx].reset_index()
            master_df = pd.concat([master_indexed.reset_index(), new_rows], ignore_index=True)
        
        print(f"Successfully merged {cell_type}.csv")
        processed += 1
    
    print(f"✅ Merged {processed} CSVs")

    psi_cols = [col for col in master_df.columns if col in all_cells_set]   # Get all PSI columns
    
    # Remove rows with no PSI values at all
    psi_vals = master_df[psi_cols].apply(pd.to_numeric, errors='coerce')
    mask_all_nan = psi_vals.isna().all(axis=1)
    num_removed = mask_all_nan.sum()
    if num_removed > 0:
        print(f"🧹 Removing {num_removed} rows with no PSI values in any cell type")
    master_df = master_df.loc[~mask_all_nan].reset_index(drop=True)

    # ---------- FORMATTING OUTPUT ----------

    # Reorder columns
    
    # Insert empty "exon_boundary" column
    master_df["exon_boundary"] = ""
    # Explicitly set order in case later pipelines require it. Will add 
    metadata_cols = [
        "cassette_exon", "alternative_splice_site_group", "linked_exons", "mutually_exclusive_exons",
        "exon_strand", "exon_length", "gene_type", "gene_id", "gene_symbol", "exon_location", "exon_boundary"
    ]
    cols_in_order = metadata_cols
    cols_in_order.extend(
                [c for c in master_df.columns if c not in metadata_cols and c != "chromosome"]
    )
    cols_in_order.append("chromosome")
    master_df = master_df[cols_in_order]

    # Construct exon_id column
    master_df.insert(
        0,
        "exon_id", 
        [f"TS_{i:06d}" for i in range(1, len(master_df)+1)]
    )

    # Calculate mean PSI
    master_df["mean_psi"] = psi_vals.mean(axis=1)
    print(f"✅ Calculated mean PSI")

    # logit mean PSI
    eps = 1e-6
    master_df["logit_mean_psi"] = logit(np.clip(master_df["mean_psi"] / 100, eps, 1-eps))
    print(f"✅ Calculated logit mean PSI")


    print(f"Columns: {master_df.columns}")
    
    output_path = os.path.join(main_dir, "psi_data", "final_data", "full_cassette_exons_with_mean_psi.csv")
    master_df.to_csv(output_path, sep=",", index=False)
    if os.path.exists(output_path):
        print(f"✅ Successfully saved full dataframe in {output_path}")
        print(f"Final output has shape {master_df.shape}")
    else:
        print(f"⚠️ Failed to save merged dataframe")


if __name__ == "__main__":
    main()
