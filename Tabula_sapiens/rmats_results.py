"""
Note: This script assumes we are only interested in PSI at the exon level, NOT the junction level.
We will calculate PSI for exon A regardless of its junction with exon B or exon C. Because of this, 
we will ignore exon boundaries (the upstream and downstream exon).

If PSI at the junction level is of concern, see /tabula_sapiens/SE_data/ and/or see the comments in this program

This script takes in --cell_type and --main_dir as command line arguments, where main_dir is the 
Tabula Sapiens directory. If no cell_type is specified, the program will look for completed.json in the main
directory and use that. 

It looks through the main directory for the zipped rMATS outputs, unzips them, and
then calculates the accumulated PSI values for each splicing event and each cell type. For now, we are only
considering skipped exon events. Finally, it deletes the unzipped output directory to save space.

Output: The code saves a CSV in ASCOT format using the information obtained from SE.MATS.JCEC.txt and saves it in
main_dir/psi_data.
"""
import pandas as pd
import numpy as np
import argparse
import json
import os
import zipfile
import shutil


def process_cell_type(cell_type, main_dir):
    # Unzip output dir
    cell_dir = os.path.join(main_dir, "rmats", cell_type)
    zip_path = os.path.join(cell_dir, "output_archive.zip")
    output_dir = os.path.join(cell_dir, "output")

    # --- Unzip only if output/ doesn't already exist ---
    if not os.path.exists(output_dir):
        print(f"\tUnzipping...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(cell_dir)
        print(f"\tUnzipped!")

    try:
        rmats_df = pd.read_csv(os.path.join(output_dir, "SE.MATS.JCEC.txt"), sep="\t")

        # ---------- CALCULATE PSI ----------
        ijc_df = rmats_df['IJC_SAMPLE_1'].str.split(',', expand=True).replace('NA', np.nan).astype(float)
        sjc_df = rmats_df['SJC_SAMPLE_1'].str.split(',', expand=True).replace('NA', np.nan).astype(float)

        # Mask out any cells that produced NA in either column
        mask = ijc_df.isna() | sjc_df.isna()
        ijc_df[mask] = np.nan
        sjc_df[mask] = np.nan

        # Raw totals per event (sum across replicates/samples)
        rmats_df['total_inc'] = np.nansum(ijc_df.values, axis=1)
        rmats_df['total_skip'] = np.nansum(sjc_df.values, axis=1)

        # For exon-level PSI
        # ---- Group by exon coordinates and sum raw counts ----
        group_cols = ['chr', 'strand', 'exonStart_0base', 'exonEnd']
        agg_counts = (
            rmats_df
            .groupby(group_cols, as_index=False)
            .agg({
                "total_inc": "sum",
                "total_skip": "sum",
                # Take first of each
                "ID": "first",
                "GeneID": "first",
                "geneSymbol": "first",
                "upstreamES": "first",
                "upstreamEE": "first",
                "downstreamES": "first",
                "downstreamEE": "first"
            })
            )


        # Calculate PSI, normalize as described in rMATS
        e = agg_counts['exonEnd'] - agg_counts['exonStart_0base']
        len_i = 99 + e.clip(upper=99) + (e-100+1).clip(lower=0)
        len_s = 99
        agg_counts['i_norm'] = agg_counts['total_inc'] / len_i
        agg_counts['s_norm'] = agg_counts['total_skip'] / len_s
        denom = agg_counts['i_norm'] + agg_counts['s_norm']
        agg_counts['psi'] = np.where(denom == 0, '', 100 * (agg_counts['i_norm'] / denom))

        final_df = agg_counts.copy()

        # Construct and save temporary TSV. Merge to form master dataset at the end
        mini_df = pd.DataFrame({
            "cassette_exon": "Yes",
            "alternative_splice_site_group": "No",
            "linked_exons": "No",
            "mutually_exclusive_exons": "No",
            "exon_strand": final_df["strand"],
            "exon_length": final_df["exonEnd"] - final_df["exonStart_0base"],
            "gene_type": "NA",
            "gene_id": final_df["GeneID"].str.strip('"'),
            "gene_symbol": final_df["geneSymbol"].str.strip('"'),
            "exon_location": final_df["chr"] + ":" + (final_df["exonStart_0base"] + 1).astype(str) + "-" + final_df["exonEnd"].astype(str),
            # Exclude exon_boundary, since we care only about exon-level inclusion. Exon boundary info will be misleading/incorrect
            # "exon_boundary": rmats_df["chr"] + ":" + (rmats_df["upstreamEE"] + 1).astype(str) + "-" + rmats_df["downstreamES"].astype(str),
            f"{cell_type.replace('_', ' ')}": final_df['psi'],
            "chromosome": final_df["chr"]
        })

        csv_name = os.path.join(f"{main_dir}", "psi_data", f"{cell_type}.csv")
        mini_df.to_csv(csv_name, sep=",", index=False)
        if os.path.exists(csv_name):
            print(f"\t✅ Successfully saved {cell_type}.csv")
            retval = 0
        else:
            print(f"\t⚠️ Failed to save {cell_type}.csv")
            retval = 1
    
    finally:
         # --- Always clean up the unzipped folder ---
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"\tRemoved unzipped files")

    
    return retval



def main():
    parser = argparse.ArgumentParser(description="Calculate PSI and save in ASCOT format")
    parser.add_argument("--cell_type", default=None, help="Run a single cell type; otherwise run all")
    parser.add_argument("--main_dir", required=True, help="File path to Tabula Sapiens directory")
    args = parser.parse_args()

    # Set working directory
    main_dir = args.main_dir
    
    # Load all completed cell types
    with open(os.path.join(main_dir, "completed.json"), "r") as f:
        all_cell_types = json.load(f)

    if args.cell_type:
        cells_to_run = [args.cell_type]
    else:
        cells_to_run = all_cell_types
    
    # Create TSV output dir if it does not exist
    files_saved = 0
    for cell_type in cells_to_run:
        print(f"🔷 Processing {cell_type}...")
        exit_code = process_cell_type(cell_type, main_dir)
        if exit_code == 0:
            files_saved += 1
    
    print(f"\nSuccessfully saved {files_saved} files")

if __name__ == "__main__":
    main()
