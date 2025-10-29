import argparse
import os
import pandas as pd

def main():
    # Parse CLI
    parser = argparse.ArgumentParser(description="Generate rMATS input files for one cell type")
    parser.add_argument("--cell_type", required=True, help="Cell type name")
    parser.add_argument("--metadata", required=True, help="Path to metadata CSV")
    parser.add_argument("--main_dir", required=True, help="Main directory for txt files")
    args = parser.parse_args()

    cwd = args.main_dir            # "/gpfs/commons/home/nkeung/tabula_sapiens"

    # Tabula Sapiens metadata
    metadata_file = args.metadata   # "/gpfs/commons/home/nkeung/tabula_sapiens/bam_paths.tsv"
    ts = pd.read_csv(metadata_file, sep="\t")
    # Set cell type
    cell_type = args.cell_type                    # "pericyte"

    # Filter only cell_types of type "pericyte"
    ts = ts[ts["cell_type"] == cell_type]
    paths = ts["bam_path"].tolist()
    num_cells = len(paths)
    print(f"{num_cells} to process for {cell_type}\n")

    print(f"Creating intermediate directories...")
    for subdir in ["", "temp", "output"]:
        dir_path = os.path.join(cwd, cell_type.replace(" ", "_"), subdir)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Ensured directory exists: {dir_path}")

    print()

    print(f"Creating input .txt file...")
    with open(f"{cwd}/{cell_type}/b1.txt", "w") as f:
        f.write(paths[0])
        for path in paths[1:]:
            f.write(f",{path}")
    if os.path.exists(f"{cwd}/{cell_type}/b1.txt"):
        print(f"✅ Successfully saved {cwd}/{cell_type}/b1.txt\n")

if __name__ == "__main__":
    main()
