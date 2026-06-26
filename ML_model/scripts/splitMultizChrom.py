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


# def compile_exon_to_chr_mapping():
#     """
#     Scans all_exon_names.txt and joins it with data in knownGene.multiz100way to get
#     the chromosome for all exons in our dataset. It saves this mapping as exon_to_chr.csv.

#     Because knownGene.multiz100way is such a large file, this function uses Polars to
#     do a lazy scan of the file and then perform filtering without loading the entire dataframe.

#     This code has already been run once, and the mapping has already been generated. 
#     This should not need to be run again, unless different exons are used in the dataset.
#     """
#     ######### Scan knownGene for ONLY the exons we need ##########
#     # Run once. Then save mapping as exons_to_chr.csv

#     with open(os.path.join(data_dir, "all_exon_names.txt"), "r") as f:
#         exon_ids = set(line.strip() for line in f if line.strip())
#     print(f"✅ Scanned exon IDs needed")

#     # Lazy scan knownGene.multiz100way.exonNuc_exon_intron_positions.csv
#     # to avoid loading 5.7 GB as a dataframe
#     large_df = (
#         pl.scan_csv(
#             os.path.join(data_dir, "knownGene.multiz100way.exonNuc_exon_intron_positions.csv"),
#             infer_schema_length=1000,
#             dtypes={
#                 "Exon Name": pl.Utf8,
#                 "Chromosome": pl.Utf8,
#                 "Species Name": pl.Utf8
#             }
#         )
#         .filter(pl.col("Species Name") == "hg38")
#         .select([
#             "Exon Name",
#             "Chromosome"
#         ])
#     ) 

#     filtered_df = (
#         large_df
#         .filter(pl.col("Exon Name").is_in(list(exon_ids)))
#         .collect(streaming=True)
#     )

#     print(f"✅ Filtered exon-to-chromosome mapping completed")

#     filtered_df.write_csv(os.path.join(output_dir, "exon_to_chr.csv"), include_header=True)
#     print(f"✅ Saved exon to chromosome map")


def set_up_metadata_files():
    """
    Initializes sets to keep track of three metadata files:
        - completed: set of all exons processed across all splits
        - split_exons: set of all exons processed in the current split
        - unexpected_exons: set of exons with IDs not found in our exon-to-chromosome mapping
    
    If these files already exist as JSON files, load them. This should only be applicable
    to completed.json and unexpected_exons.json.

    Returns: completed: set(), split_exons: set(), unexpected_exons: set()
    """

    file_names = ["completed", "split_exons", "unexpected_exons"]

    completed, split_exons, unexpected_exons = set(), set(), set()
    metadata_sets = [completed, split_exons, unexpected_exons]

    for file, s in zip(file_names, metadata_sets):
        if file == "split_exons":
            # For now, do not load and save split_exons
            # Ideally, split_exons should not need to be an intermediate file
            continue
        path = os.path.join(output_dir, "counts", f"{file}.json")

        if os.path.exists(path):
            with open(path, "r") as f:
                s.update(set(json.load(f)))

    print("Initialized and loaded metadata files...")

    return completed, split_exons, unexpected_exons


# def load_checkpoint_files():
#     """
#     Loads intermediate checkpoint files from partially completed runs.
    
#     Deprecated. Does not fit in current pipeline, but code left for 
#     reference if something terrible happens.
#     """
#     # # Load checkpoint files if they exist
#     new_val, new_test, new_train = dict(), dict(), dict()
#     new_data_paths = []

#     for split, d in zip(
#         ["val", "test", "train"],
#         [new_val, new_test, new_train]
#     ):
#         file_path = os.path.join(output_dir, "tmp", f"{split}_chr_merged_filtered_min30Views.pkl")
#         new_data_paths.append(file_path)
        
#         if os.path.exists(file_path):
#             with open(file_path, "rb") as f:
#                 saved_dict = pickle.load(f)
#                 d.update(saved_dict)
#     print("Loaded for checkpoint files...\n")

#     return new_val, new_test, new_train


def split_error_check_and_cleanup(completed, unexpected_exons, expected, split_exons_set, new_data, target_split):
    """
    Does some simple error checking (checks that all sorted exons are saved and recorded, checks that all
    exons were all successfully mapped to a chromosome).

    Then, it saves the relevant metadata files, including the list of all completed exons as a JSON
    """
    print(f"\nSimple error checking...\n")

    if len(completed) >= expected and len(split_exons_set) == expected and len(new_data) == expected:
        print("✅ All exons completed")
    else:
        print(f"❌ {len(completed)} in completed.json set. Expected at least {expected}")
        print(f"❌ {len(split_exons_set)} saved in split set")
        print(f"❌ {len(new_data)} in final dictionary")
        raise ValueError()

    if len(unexpected_exons) == 0:
        print("✅ No failed mappings")
    else:
        print(f"❌ Chromosomes not found in knownGene mapping (total: {len(unexpected_exons)})")
        print(unexpected_exons)
        raise ValueError()

    # Save metadata files
    file_names = ["completed", f"{target_split}_exons"]
    for name, s in zip(file_names, [completed, split_exons_set]):
        csv_path = os.path.join(output_dir, "counts", f"{name}.csv")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for exon in list(s):
                writer.writerow([exon])
        print(f"Saved {name}.csv")

        if name == "completed":
            json_path = os.path.join(output_dir, "counts", "completed.json")
            with open(json_path, "w") as f:
                json.dump(list(completed), f)

    print(f"✅ Saved metadata files as CSV")


def main(target_split):
    
    ##### Load exon-chromosome mapping as a CSV #####

    # The only duplicate exons are found in both chrX and chrY. Both get sorted into the training split, 
    # so converting to dict is safe and will drop the duplicate exons.
    map_df = pd.read_csv(os.path.join(output_dir, "exon_to_chr.csv"))
    exon_to_chr = dict(
        zip(map_df["Exon Name"], map_df["Chromosome"])
    )
    print("Loaded exon to chromosome mapping...")

    ##################################################

    ##### Define Chromosomes to Split #####

    chr_to_split = {
        "chr1": "val",
        "chr2": "test",
        "chr3": "test",
        "chr5": "test",
        "chr7": "val",
        "chr9": "val"
    }

    ##################################################
    

    ########## MAIN SPLITTING LOGIC ##########

    completed, split_exons_set, unexpected_exons = set_up_metadata_files()
    num_exons_found = 0
    new_data = dict()
    print(f"Beginning split:\n")

    for old_split in ["val", "test", "train"]:
        print(f"===== Processing old {old_split} split =====\n")
        
        print(f"Loading pickle file...")
        old_file = f"{old_split}_merged_filtered_min30Views.pkl"
        with open(os.path.join(old_data_path, old_file), "rb") as f:
            old_data = pickle.load(f)       # Loaded as a dict with format {exon_id, {sequences}}

        print(f"File loaded. {len(old_data)} to look through...")
        i = 0
        # Check every element
        for exon_id, sequences in old_data.items():
            i += 1
            if i % 100_000 == 0:
                print(f"\tChecking {i}-th exon")

            if exon_id in completed:
                continue
            # Map exon_id to chromosome
            try:
                chr = exon_to_chr[exon_id]
            except KeyError:
                unexpected_exons.add(exon_id)
                continue
            
            if target_split != "train" and chr_to_split.get(chr, None) == target_split:
                if exon_id in completed:
                    # Uncaught duplicate exon found!
                    print(f"⚠️ Warning: Duplicate exon found: {exon_id}")
                new_data[exon_id] = sequences
                split_exons_set.add(exon_id)
            
            elif target_split == "train" and chr not in chr_to_split and "chr" in chr:
                if exon_id in completed:
                    # Uncaught duplicate exon found!
                    print(f"⚠️ Warning: Duplicate exon found: {exon_id}")
                if "chr" not in chr:
                    raise ValueError(f"Error: Unexpected chromosome: {chr}")
                
                new_data[exon_id] = sequences
                split_exons_set.add(exon_id)

            else:
                continue
            
            completed.add(exon_id)        
            num_exons_found += 1
        
        print(f"Finished checking old {old_split} split\n")


    split_error_check_and_cleanup(completed, unexpected_exons, num_exons_found, split_exons_set, new_data, target_split)

    # Save new data split
    print(f"\nSaving new data split...")
    new_data_path = os.path.join(output_dir, f"{target_split}_chr_merged_filtered_min30Views.pkl")
    with open(new_data_path, "wb") as f:
        pickle.dump(new_data, f)
    print(f"✅ Saved file as {target_split}_chr_merged_filtered_min30Views.pkl")

    print(f"\n All done with {target_split}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    args = parser.parse_args()

    target_split = args.split

    main(target_split)
