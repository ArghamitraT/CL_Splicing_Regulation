"""
This script compares the cosine similarity scores obtained by grabbing sequences from
various sources. It grabs values stored a Python dictionary saved as a .json file.
"""

import matplotlib.pyplot as plt
import json
import os
from ete3 import Tree
import argparse

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

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gene",
    type = str,
    required = True,
    choices=["foxp2","tp53"]
)
parser.add_argument(
    "--methods",
    type = str,
    nargs = 2,
    required = True,
    choices = ['m', 'r', 'e']
)

args = parser.parse_args()
gene = args.gene
methods = args.methods

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

method_map = {
    'm': {
        'label': "Multiz MSA",
        'cos_path': input_dir + f"multiz_cos_{gene}.json",
        'seq_path': sequence_dir + f"multiz_msa/{gene}-full-stitched.json"
    },
    'r': {
        'label': "Reference Seq",
        'cos_path': input_dir + f"ref_cos_{gene}.json",
        'seq_path': sequence_dir + f"from_ref_seqs/{gene}-full-stitched.json"
    },
    'e': {
        'label': "EMBL-EBI Seq",
        'cos_path': input_dir + f"embl_cos_{gene}.json",
        'seq_path': sequence_dir + f"embl_ebi/{gene}-full-stitched.json"
    }
}

# Handle both methods being processed
method1 = method_map[methods[0]]
method2 = method_map[methods[1]]

cos1 = load_json(method1["cos_path"])
cos2 = load_json(method2["cos_path"])
seq1 = get_common_name(dict(load_json(method1["seq_path"])))
seq2 = get_common_name(dict(load_json(method2["seq_path"])))

# Exclude missing species
missing_species = []
for species in common_names:
    if cos1[species] == 0 or cos2[species] == 0:
        missing_species.append(species)

found_species = [species for species in common_names if species not in missing_species]
filtered_data1 = {species: cos1[species] for species in found_species}
filtered_data2 = {species: cos2[species] for species in found_species}

# Plotting graph
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)

x_vals = list(filtered_data1.values())
y_vals = list(filtered_data2.values())
plt.scatter(x_vals, y_vals)
plt.xlabel(f"{method1['label']} Cosine Similarity Score")
plt.ylabel(f"{method2['label']} Cosine Similarity Score")
plt.title(f"Cosine Similarities for {gene}")
for i, species in enumerate(found_species):
    if y_vals[i] / x_vals[i] > 1.5 or y_vals[i] / x_vals[i] < 0.5:
        ax.text(x_vals[i], y_vals[i], species)
min_val = min(min(x_vals), min(y_vals))
max_val = max(max(x_vals), max(y_vals))
plt.plot([min_val, max_val], [min_val, max_val], color='gray')

plt.savefig(os.path.join(output_dir, f"cos_compare_{methods[0]}_{methods[1]}_{gene}.png"), dpi=300, bbox_inches="tight")

# Calculate edit distance between the two sequences obtained
edit_distances = [levenshteinDistance(seq1[s], seq2[s]) for s in found_species]

plt.figure(figsize=(18,6))
plt.bar(found_species, edit_distances)
plt.xticks(rotation=90)
plt.xlabel("Species")
plt.ylabel("Edit Distance")
plt.title(f"Edit Distance Between {method1['label']} and {method2['label']}")
plt.savefig(os.path.join(output_dir, f"edit_dist_{methods[0]}_{methods[1]}_{gene}.png"), dpi=300, bbox_inches="tight")
