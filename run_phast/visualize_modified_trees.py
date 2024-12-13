"""
Script: visualize_modified_trees.py

Description:
This script processes a phylogenetic tree from a `.mod` file, systematically doubles the branch lengths one at a time, 
and visualizes each modified tree. For each branch, the script highlights the manipulated branch and all its descendant 
branches in red, while keeping the rest of the tree in black. The resulting visualizations are saved as PNG files.

The script handles:
1. Parsing the input `.mod` file to extract metadata and the phylogenetic tree in Newick format.
2. Iteratively modifying the branch lengths of the tree, doubling one branch at a time.
3. Saving visualizations for each modified tree with clear highlights on the manipulated branch and its descendants.

Inputs:
- `mod_file` (str): Path to the `.mod` file containing the phylogenetic tree and metadata.
  - The `.mod` file must include a line starting with `TREE:` followed by the tree in Newick format.
- `output_dir` (str): Directory to save the generated visualizations.
  - Each PNG file is named `<branch_name>_<branch_number>_doubled.png`.

Outputs:
- PNG visualizations for each modified tree. 
  - Manipulated branches and their descendants are highlighted in red.
  - Other branches are shown in black.
- The files are saved in the specified `output_dir`.

Workflow:
1. **Input Parsing**:
   - Reads and extracts metadata and tree structure from the `.mod` file.
2. **Tree Modification**:
   - Doubles the branch length for one branch at a time and resets it after saving.
3. **Visualization**:
   - Creates a circular tree visualization with branch-specific highlighting and saves it as a PNG.
4. **Output**:
   - Stores all visualizations in the specified output directory.

Dependencies:
- Requires `ete3` for tree parsing and visualization.
- Compatible with environments that support offscreen rendering (`QT_QPA_PLATFORM=offscreen`).

Usage:
Run the script directly with the default paths or specify the input file and output directory:
   python visualize_modified_trees.py

Optional arguments:
   --mod_file: Path to the input `.mod` file containing the phylogenetic tree.
   --output_dir: Directory to save the generated PNG visualizations.

Example:
   python visualize_modified_trees.py --mod_file example.mod --output_dir visualizations

Ensure all dependencies are installed, including `ete3`, and that offscreen rendering is configured for headless environments.

Environment:
    phastcon
"""


import argparse
from ete3 import Tree, TreeStyle, NodeStyle
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
# from ete3 import TreeStyle

def parse_mod_file(mod_file):
    metadata = []
    tree_str = None
    with open(mod_file, 'r') as file:
        for line in file:
            if line.startswith("TREE:"):
                tree_str = line.split("TREE:")[1].strip()
            else:
                metadata.append(line.strip())
    if not tree_str:
        raise ValueError("No TREE found in the .mod file.")
    return metadata, tree_str

def save_mod_file(metadata, tree, filename):
    with open(filename, "w") as file:
        for line in metadata:
            file.write(line + "\n")
        file.write(f"TREE: {tree.write(format=5)}\n")

def visualize_tree(tree, highlighted_node, output_file):
    # Style for highlighted branches
    highlighted_style = NodeStyle()
    highlighted_style["fgcolor"] = "red"
    highlighted_style["size"] = 0
    highlighted_style["vt_line_color"] = "red"
    highlighted_style["hz_line_color"] = "red"

    # Style for default branches
    default_style = NodeStyle()
    default_style["fgcolor"] = "black"
    default_style["size"] = 0
    default_style["vt_line_color"] = "black"
    default_style["hz_line_color"] = "black"

    # Apply styles
    for node in tree.traverse():
        node.set_style(default_style)

    for descendant in highlighted_node.iter_descendants():
        descendant.set_style(highlighted_style)
    highlighted_node.set_style(highlighted_style)

    # Tree style
    ts = TreeStyle()
    ts.show_leaf_name = True
    ts.mode = "c"  # Circular mode, change to "r" for rectangular mode

    # Render the tree
    tree.render(output_file, tree_style=ts)
    print(f"Visualization saved as {output_file}")

def modify_and_save_trees(metadata, tree, output_dir):
    branch_count = 1
    name = "root"
    for node in tree.traverse():
        if node.is_root():
            continue

        original_length = node.dist
        node.dist *= 2
        if node.name:
            name = node.name
        filename = f"{output_dir}/{name}_{branch_count}_doubled.mod"
        # save_mod_file(metadata, tree, filename)

        # Save visualization
        visualization_file = f"{output_dir}/{name}_{branch_count}_doubled.png"
        visualize_tree(tree, node, visualization_file)

        node.dist = original_length
        branch_count += 1

def main():
    mod_file_default = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/19_chr/TREES/newtrees_chr19.noncons.mod'
    output_file_default = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/19_chr/TREES/figures_circular'

    parser = argparse.ArgumentParser(description="Modify tree branch lengths.")
    parser.add_argument("--mod_file", type=str, default=mod_file_default, help="Path to the .mod file containing the tree.")
    parser.add_argument("--output_dir", type=str, default=output_file_default, help="Directory to save the modified .mod files and visualizations.")
    args = parser.parse_args()

    try:
        metadata, tree_str = parse_mod_file(args.mod_file)
    except ValueError as e:
        print(f"Error: {e}")
        return

    tree = Tree(tree_str)
    modify_and_save_trees(metadata, tree, args.output_dir)

if __name__ == "__main__":
    main()
