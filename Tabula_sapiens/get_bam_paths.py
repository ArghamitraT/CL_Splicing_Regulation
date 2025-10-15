import pandas as pd
import os

DATA_DIR = "/gpfs/commons/datasets/controlled/CZI/tabula-sapiens/TabulaSapiens_v2/SS2"
ts = pd.read_csv("/gpfs/commons/home/nkeung/tabula_sapiens/metadata.tsv", sep="\t")

# Info about dataset
print("----- TABULA SAPIENS METADATA -----\n")

print(f"Total cells: {len(ts)}")
print(f"Total patients: {len(ts['donor'].unique())}\n")

print(f"Cell Types with over 30 cells:")
counts = ts['cell_type'].value_counts()
common_cell_types = counts[counts > 30]
print(common_cell_types)
print()

# # Check if all IDs match the same pattern
# def check_id_pattern(x):
#     # Extract everything before the first dot (or adjust if dots aren't consistent)
#     return re.split(r'\.', x)[0].count('_')

def construct_path(id):
    # Split using "per"
    prefix, suffix = id.split("_per_", 1)

    # Extract donor dir
    prefix_split = prefix.split("_", 1)
    donor = prefix_split[0]
    subdir1 = prefix_split[1]

    # Exceptions for patient samples TSP20 and TSP28
    if donor == "TSP20":
        subdir1 = "TSP20_output_raw"

    elif donor == "TSP28":
        if "BlueBC6" in suffix:
            subdir1 = "TSP28_smartseq2_plate3_output_raw"
        elif "BlueBC7" in suffix:
            subdir1 = "TSP28_smartseq2_plate2_output_raw"
        elif "BlueBC8" in suffix:
            subdir1 = "TSP28_smartseq2_plate4_output_raw"
        elif "BlueBC9" in suffix: 
            subdir1 = "TSP28_smartseq2_plate1_output_raw"

    if donor == "TSP30":
        return None
    
    return f"{DATA_DIR}/{donor}/{subdir1}/per/{suffix}/Aligned.sorted.out.bam"

"""
Safe implementation of os.path.exists to account for None values (ex. for TSP30)
"""
def safe_exists(path):
    if path:
        return os.path.exists(path)
    else:
        return False

print(f"\n----------")
print("Attempting to construct file paths from IDs...\n")

ts["bam_path"] = ts["cell_id"].apply(construct_path)

# Verify that path can be found
ts["path_exists"] = ts["bam_path"].apply(safe_exists)
print(ts["path_exists"].value_counts())
print()

# Save results
ts.to_csv("/gpfs/commons/home/nkeung/tabula_sapiens/bam_paths.tsv", sep="\t", index=False)

print(f"⚠️ Warning: Missing data paths!")
missing = ts[~ts['path_exists']]
print(f"Patient samples with missing paths:")
print(missing["donor"].unique())
