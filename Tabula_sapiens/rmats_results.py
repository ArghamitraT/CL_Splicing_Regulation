import pandas as pd
import numpy as np
import argparse
import json
import os
import zipfile


# """
# Calcaulates PSI for each single cell. Should match "IncLevel1"
# """
# def per_replicate_psi(row):
#     ijc_vals = row['IJC_SAMPLE_1'].split(',')
#     sjc_vals = row['SJC_SAMPLE_1'].split(',')
#     psi_list = []
    
#     for i, s in zip(ijc_vals, sjc_vals):
#         if i == 'NA' or s == 'NA':
#             psi_list.append(np.nan)
#             continue
        
#         e = row['exonEnd'] - row['exonStart_0base']
#         len_i = 99 + min(e, 99) + max(0, e - 100 + 1)
#         len_s = 99
#         i_norm = float(i) / len_i
#         s_norm = float(s) / len_s
#         denom = i_norm + s_norm
#         psi_list.append(np.nan if denom == 0 else i_norm / denom)
    
#     # Return as comma-separated string or list, depending on what you need
#     return ','.join(f'{p:.3f}' if not np.isnan(p) else 'NA' for p in psi_list)

# def check_match(row):
#     psi_vals = row['PSI_per_replicate'].split(',')
#     inc_vals = row['IncLevel1'].split(',')
    
#     # Must have same number of replicates
#     if len(psi_vals) != len(inc_vals):
#         return False
    
#     for psi, inc in zip(psi_vals, inc_vals):
#         if psi == 'NA' and inc == 'NA':
#             continue
#         if psi == 'NA' or inc == 'NA':
#             return False
        
#         # Convert to float and allow for small rounding tolerance
#         if not np.isclose(float(psi), float(inc), atol=1e-3):
#             return False
#     return True


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

        rmats_df = pd.read_csv(os.path.join(main_dir, "rmats", f"{cell_type}", "output", "SE.MATS.JCEC.txt"), sep="\t")

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
            print(f"✅ Successfully saved {cell_type}.csv")
        else:
            print(f"⚠️ Failed to save {cell_type}.csv")
        files_saved += 1
    
    print(f"Successfully saved {files_saved} files")

if __name__ == "__main__":
    main()
