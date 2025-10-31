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

        total_inc = np.nansum(ijc_df.values, axis=1)
        total_skip = np.nansum(sjc_df.values, axis=1)

        # Calculate PSI, normalize as described in rMATS
        e = rmats_df['exonEnd'] - rmats_df['exonStart_0base']
        len_i = 99 + e.clip(upper=99) + (e-100+1).clip(lower=0)
        len_s = 99
        i_norm = total_inc / len_i
        s_norm = total_skip / len_s
        denom = i_norm + s_norm
        psi = np.where(denom == 0, '', 100 * (i_norm / denom))

        # Construct and save temporary TSV. Merge to form master dataset at the end
        mini_df = pd.DataFrame({
            "cassette_exon": "Yes",
            "alternative_splice_site_group": "No",
            "linked_exons": "No",
            "mutually_exclusive_exons": "No",
            "exon_strand": rmats_df["strand"],
            "exon_length": rmats_df["exonEnd"] - rmats_df["exonStart_0base"],
            "gene_type": "NA",
            "gene_id": rmats_df["GeneID"].str.strip('"'),
            "gene_symbol": rmats_df["geneSymbol"].str.strip('"'),
            "exon_location": rmats_df["chr"] + ":" + (rmats_df["exonStart_0base"] + 1).astype(str) + "-" + rmats_df["exonEnd"].astype(str),
            "exon_boundary": rmats_df["chr"] + ":" + (rmats_df["upstreamEE"] + 1).astype(str) + "-" + rmats_df["downstreamES"].astype(str),
            f"{cell_type.replace('_', ' ')}": psi,
            "chromosome": rmats_df["chr"]
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
