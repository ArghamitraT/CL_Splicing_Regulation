import matplotlib.pyplot as plt
import json
import os
from ete3 import Tree
import pandas as pd
import pickle

# Set offscreen rendering for ete3
os.environ["QT_QPA_PLATFORM"] = "offscreen"

input_dir = "/gpfs/commons/home/nkeung/data/"
sequence_dir = "/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/"
output_dir = "/gpfs/commons/home/nkeung/data/figures/"

# To maintain consistent ordering in similarity scores
common_tree = Tree(input_dir + "hg38.100way.commonNames.nh")
common_names = [leaf.name for leaf in common_tree.iter_leaves()]
ucsc_tree = Tree(input_dir + "hg38.100way.nh")
ucsc_codes = [leaf.name for leaf in ucsc_tree.iter_leaves()]

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def get_common_name(embeddings: dict):
    global ucsc_codes, common_names
    new_names = dict(zip(ucsc_codes, common_names))
    new_dict = {new_names[species]: embeddings[species] for species in embeddings if species in new_names}
    return new_dict

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
    
def get_exon_number(code: str):
    parts = code.split('_')
    return int(parts[1])



### ----- GET NUCLEOTIDE DICTIONARIES ----- ###
# multiz_nuc = {}
# df = pd.read_csv("/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/multiz_msa/foxp2-nuc-seqs.csv")
# df = df.sort_values(by=["Species", "Number"])
# grouped_df = df.groupby("Species")

# for species, groups in grouped_df:
#     sequence = ""
#     for exon in groups["Seq"]:
#         exon = exon.strip().replace("-", "").replace("\n", "")
#         sequence += exon
#     if len(sequence) != 0:
#         multiz_nuc[species] = sequence

# with open("/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/multiz_msa/foxp2-nuc.json", "w") as f:
#     json.dump(multiz_nuc, f)

# # Get nucleotides from pickle file
# with open("/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/from_ref_seqs/exon_nuc_seq.pkl", "rb") as f:
#     data = pickle.load(f)        # data[exon code][species] = string

# filtered_data = {key:value for (key, value) in data.items() if "ENST00000350908" in key}

# # Reshape dictionary to group by species
# grouped_exons = {}
# # for exon_code, species_dict in sorted_exons.items():
# for exon_code, species_dict in filtered_data.items():
#     for species, sequence in species_dict.items():
#         if species not in grouped_exons:
#             grouped_exons[species] = {}
#         grouped_exons[species][get_exon_number(exon_code)] = sequence

# ref_nucs = {}
# for species in grouped_exons:
#     seq = ""
#     for i in range(1, 17):
#         if i not in grouped_exons[species]:
#             continue
#         sequence = grouped_exons[species][i]
#         if sequence in [None, (None, None), ""]:
#             continue
#         seq += sequence.strip().replace("-", "").replace("\n", "")
#     ref_nucs[species] = seq
# with open("/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/from_ref_seqs/foxp2-nuc.json", "w") as f:
#     json.dump(ref_nucs, f)

with open("/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/multiz_msa/foxp2-nuc.json", "r") as f:
    multiz_nuc = json.load(f)

with open("/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/from_ref_seqs/foxp2-nuc.json", "r") as f:
    ref_nuc = json.load(f)

# Filter and clean data
multiz_nuc = get_common_name(multiz_nuc)
ref_nuc = get_common_name(ref_nuc)

missing_species = []
for species in common_names:
    if species not in multiz_nuc or species not in ref_nuc or multiz_nuc[species] == "" or ref_nuc[species] == "":
        missing_species.append(species)

filtered_species = [s for s in common_names if s not in missing_species]
filtered_multiz = {sp: multiz_nuc[sp] for sp in filtered_species}
filtered_ref = {sp: ref_nuc[sp] for sp in filtered_species}

edit_dist = [levenshteinDistance(filtered_multiz[sp], filtered_ref[sp]) for sp in filtered_species]
plt.figure(figsize=(18, 6))
plt.bar(filtered_species, edit_dist)
plt.xticks(rotation=90)
plt.xlabel("Species")
plt.ylabel("Edit Distance")
plt.title("Nucleotide Edit Distance Between Multiz and Reference Sequence (FOXP2)")
plt.savefig("/gpfs/commons/home/nkeung/data/figures/nuc_edit_dist.png", dpi=300, bbox_inches="tight")
