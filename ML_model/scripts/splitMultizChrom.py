import os
import csv
import pandas as pd
import polars as pl
import json
import pickle
import argparse

data_dir = "/mnt/home/nlk2136/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/"
old_data_path = os.path.join(data_dir, "trainTestVal_data")

output_dir = "/mnt/home/nlk2136/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/chromSplit"



########## Scan knownGene for ONLY the exons we need ##########
# Run once. Then save mapping as exons_to_chr.csv

# with open(os.path.join(data_dir, "all_exon_names.txt"), "r") as f:
#     exon_ids = set(line.strip() for line in f if line.strip())
# print(f"✅ Scanned exon IDs needed")

# # Lazy scan knownGene.multiz100way.exonNuc_exon_intron_positions.csv
# # to avoid loading 5.7 GB as a dataframe
# large_df = (
#     pl.scan_csv(
#         os.path.join(data_dir, "knownGene.multiz100way.exonNuc_exon_intron_positions.csv"),
#         infer_schema_length=1000,
#         dtypes={
#             "Exon Name": pl.Utf8,
#             "Chromosome": pl.Utf8,
#             "Species Name": pl.Utf8
#         }
#     )
#     .filter(pl.col("Species Name") == "hg38")
#     .select([
#         "Exon Name",
#         "Chromosome"
#     ])
# ) 

# filtered_df = (
#     large_df
#     .filter(pl.col("Exon Name").is_in(list(exon_ids)))
#     .collect(streaming=True)
# )

# print(f"✅ Filtered exon-to-chromosome mapping completed")

# filtered_df.write_csv(os.path.join(output_dir, "exon_to_chr.csv"), include_header=True)
# print(f"✅ Saved exon to chromosome map")


##### Load CSV and drop any duplicate exons #####

# The only duplicate exons are found in both chrX and chrY. Both get sorted into the training split, 
# so converting to dict is safe.

map_df = pd.read_csv(os.path.join(output_dir, "exon_to_chr.csv"))

exon_to_chr = dict(
    zip(map_df["Exon Name"], map_df["Chromosome"])
)
print("Loaded exon to chromosome mapping...")

######################################################


##### Define Chromosomes to Split #####

chr_to_split = {
    "chr1": "val",
    "chr2": "test",
    "chr3": "test",
    "chr5": "test",
    "chr7": "val",
    "chr9": "val"
}


# Load metadata and progress files

file_names = ["completed", "train_exons", "val_exons", "test_exons", "failed_mappings"]
progress_paths = []

completed, train_exons, val_exons, test_exons, failed_mappings = set(), set(), set(), set(), set()
metadata_sets = [completed, train_exons, val_exons, test_exons, failed_mappings]

for file, s in zip(file_names, metadata_sets):
    path = os.path.join(output_dir, "tmp", f"{file}.json")
    progress_paths.append(path)

    if os.path.exists(path):
        with open(path, "r") as f:
            s.update(set(json.load(f)))
print("Checked metadata files...")


# # Load checkpoint files if they exist
new_val, new_test, new_train = dict(), dict(), dict()
new_data_paths = []

for split, d in zip(
    ["val", "test", "train"],
    [new_val, new_test, new_train]
):
    file_path = os.path.join(output_dir, "tmp", f"{split}_chr_merged_filtered_min30Views.pkl")
    new_data_paths.append(file_path)
    
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            saved_dict = pickle.load(f)
            d.update(saved_dict)
print("Looked for checkpoint files...\n")

for split, new_dict in zip(["val", "test", "train"], [new_val, new_test, new_train]):
    print(f"===== Starting {split} split =====")
    
    old_file = f"{split}_merged_filtered_min30Views.pkl"
    with open(os.path.join(old_data_path, old_file), "rb") as f:
        old_data = pickle.load(f)       # Loaded as a dict with format {exon_id, {sequences}}
    print(f"{len(old_data)} exons in file...")

    already_done = len(completed)

    i = 0
    # Process every single element
    for exon_id, sequences in old_data.items():
        if exon_id in completed:
            continue
        # Map exon_id to chromosome
        try:
            chr = exon_to_chr[exon_id]
        except KeyError:
            failed_mappings.add(exon_id)
            continue
        
        if chr not in chr_to_split and "chr" in chr:
            new_train[exon_id] = sequences
            train_exons.add(exon_id)

        elif chr_to_split[chr] == "val":
            new_val[exon_id] = sequences
            val_exons.add(exon_id)

        elif chr_to_split[chr] == "test":
            new_test[exon_id] = sequences
            test_exons.add(exon_id)

        else:
            raise ValueError(f"Error: Unexpected chromosome: {chr}")
        
        completed.add(exon_id)
        i += 1
        if i % 100_000 == 0:
            print(f"\tProcessed {i} exons")

    if len(completed) == already_done:
        print("No work done, no need to update checkpoints")
        print("===== End of split =====")
        continue
    
    print(f"🔹 Successfully processed old {split} split")
    
    # Save metadata files
    for path, s in zip(progress_paths, metadata_sets):
        with open(path, "w") as f:
            json.dump(list(s), f)
    print(f"\tSaved temporary metadata files")

    # Checkpoint data
    for new_data, path in zip([new_val, new_test, new_train], new_data_paths):
        with open(path, "wb") as f:
            pickle.dump(new_data, f)
    print(f"\tSaved checkpoint data")

    # Delete old data to free up memory
    del old_data
    gc.collect()

    print("===== End of split =====\n")


##### CLEANUP #####
def cleanup():
    print(f"\nMinor error checking...\n")

    if len(completed) == 924_629 and len(val_exons) + len(test_exons) + len(train_exons) == 924_629:
        print("✅ All exons completed")
    else:
        print(f"❌ {len(completed)} in completed.json set. Expected 924,629")
        print(f"❌ {len(val_exons) + len(test_exons) + len(train_exons)} found in dicts")
        raise ValueError()

    if len(failed_mappings) == 0:
        print("✅ No failed mappings")
    else:
        print(f"❌ Chromosomes not found in knownGene mapping (total: {len(failed_mappings)})")
        print(failed_mappings)
        raise ValueError()

    # Save completed sets as CSVs
    # Save metadata files
    for name, s in zip(file_names, metadata_sets):
        if name == "failed_mappings":
            continue
        final_path = os.path.join(output_dir, f"{name}.csv")

        with open(final_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for exon in list(s):
                writer.writerow([exon])
        print(f"Saved {name}.csv")

    print(f"Saved metadata files as CSV")


def main(target_split):
    
    # set_up_metadata_files()

    print("\n🥹 All Done!")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--split", choices=["train", "val", "test"], required=True)
#     args = parser.parse_args()

#     target_split = args.split
    
#     main(target_split)
