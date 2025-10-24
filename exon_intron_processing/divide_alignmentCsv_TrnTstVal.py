import pandas as pd
import glob
import os

# === File paths ===

division = 'train'

MASTER_DIR = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/exon_intron_positions_5and3prime/'
MASTER_PATTERN = os.path.join(MASTER_DIR, "alignmentknownGene.multiz100way.exonNuc_exon_intron_positions_split_file_*_adjustedIntron300bpOffset.csv")
DATA_LIST = f"/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/{division}_exon_list.csv"
# VAL_LIST   = "val_exon_list.csv"
# TEST_LIST  = "test_exon_list.csv"


# === Load exon lists ===
def load_exon_ids(path):
    return set(pd.read_csv(path)["exon_id"].astype(str))

data_ids = load_exon_ids(DATA_LIST)
    # val_ids   = load_exon_ids(VAL_LIST)
    # test_ids  = load_exon_ids(TEST_LIST)

print(f"Loaded {len(data_ids)} data exon IDs")
# === Prepare cumulative output containers ===
data_dfs = []
total_common_exons = 0

# === Iterate through each master file ===
for master_file in sorted(glob.glob(MASTER_PATTERN)):
    print(f"Processing: {os.path.basename(master_file)}")
    df = pd.read_csv(master_file)
    
    # Convert Exon_Name to string
    df["Exon_Name"] = df["Exon_Name"].astype(str)
    
    # Split subsets
    # train_df = df[df["Exon_Name"].isin(train_ids)]
    # val_df   = df[df["Exon_Name"].isin(val_ids)]
    # test_df  = df[df["Exon_Name"].isin(test_ids)]
    data_df = df[df["Exon_Name"].isin(data_ids)]
    
    # Append to accumulators
    data_dfs.append(data_df)

    print(f"  → total rows with homologs: {len(data_df)}")
    common_exons = len(set(data_df["Exon_Name"]))
    total_common_exons += common_exons
    print(f"  → common exons: {common_exons}")

# === Concatenate and save results ===
output_path = os.path.join(MASTER_DIR, f"alignment_file_{division}.csv")
pd.concat(data_dfs).to_csv(output_path, index=False)
# pd.concat(val_dfs).to_csv(os.path.join(OUT_DIR, "alignment_val.csv"), index=False)
# pd.concat(test_dfs).to_csv(os.path.join(OUT_DIR, "alignment_test.csv"), index=False)

print(f"\nTotal common exons across all files: {total_common_exons}")
print(f"\n✅ Split complete! Saved under: {output_path}")