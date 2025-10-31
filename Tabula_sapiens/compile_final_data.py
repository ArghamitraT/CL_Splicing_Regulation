import pandas as pd
from functools import reduce
import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Combine intermediate tables to final data")
    parser.add_argument("--main_dir", required=True, help="File path to Tabula Sapiens directory")
    args = parser.parse_args()

    main_dir = args.main_dir
    
    with open(os.path.join(main_dir, "completed.json"), "r") as f:
        all_cell_types = list(json.load(f))
    all_cell_types = ["pericyte", "mesenchymal_stem_cell_of_adipose_tissue", "ltf+_epithelial_cell"]    # Test set chosen at random
    all_cells_set = set([name.replace("_", " ") for name in all_cell_types])

    csv_dir = os.path.join(main_dir, "psi_data")

    # Columns to merge on
    merge_keys = ["gene_id", "exon_location", "exon_boundary", "exon_strand", "chromosome"]

    # Load all dataframes at once
    df_list = []
    for cell_type in all_cell_types:
        csv_file = os.path.join(csv_dir, f"{cell_type}.csv")
        df = pd.read_csv(csv_file, sep=",")
        df_list.append(df)

    # Merge all in one shot
    master_df = reduce(
    lambda left, right: pd.merge(left, right, on=merge_keys, how="outer"),
    df_list
    )

    # Construct exon_id column
    master_df.insert(
        0,
        "exon_id", 
        [f"TS_{i:06d}" for i in range(1, len(master_df)+1)]
    )

    # Calculate mean PSI
    psi_cols = [col for col in master_df.columns if col in all_cells_set]   # Get all PSI columns

    psi_vals = master_df[psi_cols].apply(pd.to_numeric, errors='coerce')
    master_df["mean_psi"] = psi_vals.mean(axis=1)
    
    output_path = os.path.join(main_dir, "final_data", "subset_cassette_exons_with_mean_psi.csv")
    master_df.to_csv(output_path, sep=",")
    if os.path.exists(output_path):
        print(f"✅ Successfully saved full dataframe in {output_path}")
        print(f"Final output has shape {master_df.shape}")
    else:
        print(f"⚠️ Failed to save merged dataframe")


if __name__ == "__main__":
    main()
