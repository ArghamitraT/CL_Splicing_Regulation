import pickle

# --- File paths ---
# split_name = "val"
# main_dir = "/mnt/home/at3836/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/"
# path_5p = main_dir + f"{split_name}_5primeIntron_filtered.pkl"
# path_3p = main_dir + f"{split_name}_3primeIntron_filtered.pkl"
# path_exon = main_dir + f"{split_name}_ExonSeq_filtered.pkl"
# output_path = main_dir + f"{split_name}_merged_filtered_min30Views.pkl"

split_name = "train"
main_dir = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/"
path_5p = main_dir + f"{split_name}_ASCOT_5primeIntron_filtered.pkl"
path_3p = main_dir + f"{split_name}_ASCOT_3primeIntron_filtered.pkl"
path_exon = main_dir + f"{split_name}_ASCOT_ExonSeq_filtered.pkl"
output_path = main_dir + f"{split_name}_ASCOT_merged_filtered_min30Views.pkl"


# --- Load files ---
with open(path_5p, "rb") as f:
    data_5p = pickle.load(f)

with open(path_3p, "rb") as f:
    data_3p = pickle.load(f)

with open(path_exon, "rb") as f:
    data_exon = pickle.load(f)

# --- Merge and filter ---
merged_data = {}
i = 0

def _process_exon_removeN(exon_seq):
    return exon_seq.replace("N", "")

for exon_id in data_5p:
    if exon_id not in data_3p or exon_id not in data_exon:
        continue

    species_5p = set(data_5p[exon_id].keys())
    species_3p = set(data_3p[exon_id].keys())
    species_exon = set(data_exon[exon_id].keys())
    
    common_species = species_5p & species_3p & species_exon

    if len(common_species) < 30:
        continue

    merged_data[exon_id] = {}

    # for species in common_species:
    #     full_seq = (
    #         data_5p[exon_id][species] +
    #         data_exon[exon_id][species]+
    #         data_3p[exon_id][species]
    #     )
    #     merged_data[exon_id][species] = full_seq
    
    for species in common_species:
        merged_data[exon_id][species] = {
            "5p": data_5p[exon_id][species],
            "exon": _process_exon_removeN(data_exon[exon_id][species]),
            "3p": data_3p[exon_id][species]
        }

    i += 1
    print(i)


# --- Save output ---
with open(output_path, "wb") as f:
    pickle.dump(merged_data, f)

print(f"Saved merged data with ≥30 species per exon to: {output_path}")
print(f"exon num: {i}")
