import argparse
import os
from ete3 import Tree
import json
import matplotlib.pyplot as plt

# Set offscreen rendering for ete3
os.environ["QT_QPA_PLATFORM"] = "offscreen"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gene",
    type = str,
    required = True,
    default = "foxp2",
    choices = ["foxp2", "tp53"]
)
args = parser.parse_args()
gene = args.gene

input_dir = f"/gpfs/commons/home/nkeung/data/"
cos_file = os.path.join(input_dir, f"multiz_cos_{gene}.json")
common_tree = Tree(os.path.join(input_dir, f"hg38.100way.commonNames.nh"))
output_dir = f"/gpfs/commons/home/nkeung/data/figures/"

with open(cos_file, "r") as f:
    cos_scores = json.load(f)

common_names = [leaf.name for leaf in common_tree.iter_leaves()]

found_species = [sp for sp in common_names if sp in cos_scores]

phylo_dist = []
for species in found_species:
    node = common_tree.search_nodes(name=species)[0]
    phylo_dist.append(node.get_distance("Human"))
    
fig = plt.figure(figsize=(18,6))
ax = fig.add_subplot(111)
plt.scatter(cos_scores.values(), phylo_dist)
# for i, species in enumerate(found_species):
#     if species in ["Gorilla", "Rhesus"]:
#         ax.text(list(cos_scores.values())[i], phylo_dist[i], species)
plt.xlabel("Cosine Similarity")
plt.ylabel("Phylogenetic Distance")
plt.title(f"Embedding Similarity and Phylogenetic Distance for {gene}")
plt.savefig(os.path.join(output_dir, f"phylo_dist_cos_{gene}.png"), dpi=300, bbox_inches="tight")
