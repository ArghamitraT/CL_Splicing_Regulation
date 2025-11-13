import os
from ete3 import Tree, TreeStyle, NodeStyle, TextFace

os.environ["QT_QPA_PLATFORM"] = "offscreen"
common_tree = Tree("/gpfs/commons/home/nkeung/data/hg38.100way.commonNames.nh")
ts = TreeStyle()
ts.show_leaf_name = True
ts.mode = "c"
common_tree.render("/gpfs/commons/home/nkeung/Contrastive_Learning/code/ML_model/figures/recomb_2026/phylo_tree.svg", tree_style=ts)