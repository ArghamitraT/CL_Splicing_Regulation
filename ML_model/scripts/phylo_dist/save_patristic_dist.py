from ete3 import Tree
import pickle
import pandas as pd

tree_path = f"/gpfs/commons/home/nkeung/data/hg38.100way.nh"
t = Tree(tree_path)

leaves = t.get_leaves()

# Iterate through and get all species names
all_species = [leaf.name for leaf in leaves]

dist_matrix = pd.DataFrame(index=all_species, columns=all_species, dtype=float)

for i, node1 in enumerate(leaves):
    for j, node2 in enumerate(leaves):
        # Matrix is symmetric
        if i <= j:
            dist_matrix.loc[node1.name, node2.name] = node1.get_distance(node2)
            dist_matrix.loc[node2.name, node1.name] = node1.get_distance(node2)

print(dist_matrix)

dist_matrix.to_csv("/gpfs/commons/home/nkeung/Contrastive_Learning/data/phylo_dist/species_branch_dist.csv")
with open("/gpfs/commons/home/nkeung/Contrastive_Learning/data/phylo_dist/species_branch_dist.pkl", "wb") as f:
    pickle.dump(dist_matrix, f)
