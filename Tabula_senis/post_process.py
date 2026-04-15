import os
import shutil
import json
import argparse

# Arguments:
# --main_dir /gpfs/commons/home/nkeung/tabula_sapiens

def main():
    parser = argparse.ArgumentParser(description="Delete temp files saved in --main_dir and --cell_type, log rMATS completion.")
    parser.add_argument("--cell_type", required=True, help="Cell type name")
    parser.add_argument("--main_dir", required=True, help="Main directory for txt files")
    args = parser.parse_args()

    cell_type = args.cell_type
    main_dir = args.main_dir
    cell_dir = os.path.join(main_dir, cell_type.replace(" ", "_"))

    # Delete temp files
    dirs_to_delete = [
        os.path.join(cell_dir, "temp"),
        os.path.join(cell_dir, "output", "tmp"),
        os.path.join(cell_dir, "data")
    ]
    for d in dirs_to_delete:
        if os.path.exists(d):
            shutil.rmtree(d)  # Recursively delete all contents
            print(f"Deleted {d}")
        else:
            print(f"Skipped {d} (does not exist)")
    print()

    # Zip output files
    output_dir = os.path.join(cell_dir, "output")
    if os.path.exists(output_dir):
        zip_path = os.path.join(cell_dir, "output_archive")
        shutil.make_archive(base_name=zip_path, format="zip", root_dir=cell_dir, base_dir="output")
        print(f"Zipped output directory to {zip_path}.zip")
    else:
        print(f"Output directory {output_dir} not found.")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Removed unzipped output folder {output_dir}")

    # Add cell type to completed JSON
    json_file = os.path.join(main_dir, "completed.json")
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            completed_cells = set(json.load(f))
    else:
        completed_cells = set()
    
    completed_cells.add(cell_type.replace(" ", "_"))

    with open(json_file, "w") as f:
        json.dump(list(completed_cells), f)


if __name__ == "__main__":
    main()
