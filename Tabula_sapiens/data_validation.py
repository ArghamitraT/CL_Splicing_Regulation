import pandas as pd
import numpy as np
import json

def sanity_checks(full_df, psi_cols):
    problems = 0
    # 1. Uniqueness of exon_ids
    duplicates = full_df[full_df['exon_id'].duplicated(keep=False)]
    if not duplicates.empty:
        print("❌ Duplicate exon_ids found:")
        print(duplicates)
        problems += 1
    else:
        print("✅ exon_id is unique")


    # 2. Exon Boundary should be empty (type is float64)
    if (full_df["exon_boundary"].isna()).all():
        print("✅ All exon_boundary values are empty")
    else:
        print(f"❌ Some exon_boundary values are not empty")
        problems += 1


    # 3. Ensure exon_location has a value
    missing_location = full_df["exon_location"].isna().sum() + (full_df["exon_location"] == "").sum()
    if missing_location > 0:
        print(f"❌ Found {missing_location} rows with missing exon_location")
        problems += 1
    else:
        print("✅ exon_location is filled")


    # 4. PSI columns: either numeric or empty string
    for col in psi_cols:
        invalid_vals = full_df[~full_df[col].apply(lambda x: isinstance(x, (int, float)) or x == "")][col]
        if len(invalid_vals) > 0:
            print(f"❌ Column {col} has {len(invalid_vals)} invalid values")
            problems += 1

    if len(psi_cols) > 0:
        print("✅ Checked PSI columns for valid values")

    # 5. All rows must have at least one PSI value
    psi_numeric = full_df[psi_cols].apply(pd.to_numeric, errors='coerce')

    # Mask: True if row has **all NaN** in numeric form (i.e., no numeric PSI at all)
    mask_all_nan = psi_numeric.isna().all(axis=1)

    num_invalid = mask_all_nan.sum()
    if num_invalid > 0:
        print(f"❌ Found {num_invalid} rows with no numeric PSI values")
        # Print row indices and PSI values for inspection
        print(full_df.loc[mask_all_nan, psi_cols])
        problems += 1
    else:
        print(f"✅ All rows have at least one numeric PSI value")


    # # 5.5 For training/validation/test sets, no NAN values allowed. Must be -1
    # psi_numeric = full_df[psi_cols].apply(pd.to_numeric, errors='coerce')

    # # Mask: All rows where PSI is not NA and non-negative OR PSI is -1
    # mask_invalid = ~( (psi_numeric.notna()) & ((psi_numeric >= 0) | (psi_numeric == -1)) ).any(axis=1)

    # num_invalid = mask_invalid.sum()
    # if num_invalid > 0:
    #     print(f"❌ Found {num_invalid} rows where all PSI values are invalid (NaN or < -1)")
    #     # Optionally print these rows for inspection
    #     print(full_df.loc[mask_invalid, psi_cols])
    # else:
    #     print(f"✅ All rows have at least one valid PSI value (-1 or >=0)")


    # 6. Mean PSI and Logit Mean PSI should not be NAN or empty
    for col in ["mean_psi", "logit_mean_psi"]:
        nan_count = full_df[col].isna().sum()
        empty_count = (full_df[col] == "").sum()
        if nan_count + empty_count > 0:
            print(f"❌ Column {col} has {nan_count} NaN values and {empty_count} \"\" values")
            problems += 1
        else:
            print(f"✅ Column {col} has no NaN values")

    if problems == 0:
        print(f"\n✅ All checks passed!")
    else:
        print(f"\n{problems} problems left to address!")



def main():
    full_ascot_file = "/gpfs/commons/home/nkeung/tabula_sapiens/psi_data/final_data/full_cassette_exons_with_mean_psi.csv"
    # full_ascot_file = "/gpfs/commons/home/nkeung/tabula_sapiens/psi_data/final_data/val_cassette_exons.csv"

    full_df = pd.read_csv(full_ascot_file)

    with open("/gpfs/commons/home/nkeung/tabula_sapiens/completed.json", "r") as f:
        cells = list(json.load(f))
        psi_cols = [x.replace("_", " ") for x in cells]
    # psi_cols = ["pericyte", "mesenchymal stem cell of adipose tissue", "ltf+ epithelial cell"]    # For subset exons

    sanity_checks(full_df, psi_cols)
    
if __name__ == "__main__":
    main()
