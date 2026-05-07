import pandas as pd
import json
import os

if __name__ == "__main__":
    
    # Load metadata and completed.json
    DIR_NAME = f"/gpfs/commons/home/nkeung/tabula_sapiens/"
    ts_metadata = os.path.join(DIR_NAME, "cell_counts.tsv")
    json_file = os.path.join(DIR_NAME, "completed.json")
    
    ts_df = pd.read_csv(ts_metadata, sep="\t")
    with open(json_file) as f:
        completed_cells = set(json.load(f))

    full_cell_classes = list(ts_df["cell_type"].unique())
    
    missing = []
    for cell in full_cell_classes:
        cell_name = cell.replace(" ", "_")
        if cell_name not in completed_cells:
            missing.append(cell)

    if missing:
        raise RuntimeError(f"Missing {len(missing)} cells: {missing}")
    else:
        print(f"✅ All cell classes have rMATS results")
