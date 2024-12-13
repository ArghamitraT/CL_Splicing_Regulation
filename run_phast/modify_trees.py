"""
Script: modify_trees.py

Description:
This script takes a phylogenetic tree from a `.mod` file, doubles the length of each branch one at a time, and saves 
the modified trees as new `.mod` files. Each output file retains the metadata from the original `.mod` file 
and includes the updated tree with one branch length modified.

Inputs:
1. `mod_file` (str): Path to the `.mod` file containing the phylogenetic tree and metadata.
   - The `.mod` file must have a line starting with `TREE:` followed by the tree in Newick format.
2. `output_dir` (str): Directory to save the modified `.mod` files.
   - If not specified, default paths are used for both `mod_file` and `output_dir`.

Outputs:
- One `.mod` file for each branch of the tree, with its length doubled. The output files are named as:
  `<branch_name>_<branch_number>_doubled.mod`
  Example: `mouse_2_doubled.mod`

What the Script Does:
1. Parses the input `.mod` file to extract metadata and the tree.
2. Traverses each branch of the tree:
   - Doubles the branch length.
   - Saves the modified tree and metadata into a new `.mod` file.
   - Resets the branch length to its original value before proceeding to the next branch.
3. Saves all modified `.mod` files into the specified output directory.

Dependencies:
- Python library `ete3` is required for tree parsing and manipulation.
- The script uses `argparse` to handle command-line arguments.

Usage:
Run the script with:
   python modify_trees.py --mod_file <path_to_mod_file> --output_dir <output_directory>
Or, rely on default paths:
   python modify_trees.py

Example:
   python modify_trees.py --mod_file example.mod --output_dir modified_trees

Environment:
    phastcon
"""


import argparse
from ete3 import Tree

def parse_mod_file(mod_file):
    """
    Parse the tree and metadata from a .mod file.
    """
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
    """
    Save the tree and metadata to a .mod file.
    """
    with open(filename, "w") as file:
        # Write metadata
        for line in metadata:
            file.write(line + "\n")
        # Write the updated tree
        file.write(f"TREE: {tree.write(format=5)}\n")

def modify_and_save_trees(metadata, tree, output_dir):
    """
    Modify the branch lengths of the tree by doubling them one at a time.
    Save the modified trees as .mod files.
    """
    branch_count = 1
    name = 'root'
    for node in tree.traverse():
        if node.is_root():
            continue  # Skip the root node as it doesn't have a branch length

        original_length = node.dist
        node.dist *= 2  # Double the branch length
        if node.name:
            name = node.name
        # Save the modified tree
        filename = f"{output_dir}/{name}_{branch_count}_doubled.mod"
        save_mod_file(metadata, tree, filename)
        print(f"Saved tree with branch {branch_count} doubled to {filename}")

        # Reset the branch length to the original value for the next iteration
        node.dist = original_length
        branch_count += 1

def main():
    # Default paths
    mod_file_default = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/dummy/mytrees.noncons.mod'
    output_file_default = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/dummy/modified_trees'

    # Argument parser
    parser = argparse.ArgumentParser(description="Modify tree branch lengths.")
    parser.add_argument("--mod_file", type=str, default=mod_file_default, help="Path to the .mod file containing the tree.")
    parser.add_argument("--output_dir", type=str, default=output_file_default, help="Directory to save the modified .mod files.")
    args = parser.parse_args()

    # Parse the tree and metadata from the .mod file
    try:
        metadata, tree_str = parse_mod_file(args.mod_file)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Load the tree using ete3
    tree = Tree(tree_str)

    # Modify and save the trees
    modify_and_save_trees(metadata, tree, args.output_dir)

if __name__ == "__main__":
    main()
