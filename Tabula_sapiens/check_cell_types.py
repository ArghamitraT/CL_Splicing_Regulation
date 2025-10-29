import pandas as pd

metadata_path = "/gpfs/commons/home/nkeung/tabula_sapiens/bam_paths.tsv"

ts = pd.read_csv(metadata_path, sep="\t")

cell_counts = ts['cell_type'].value_counts().reset_index()
cell_counts.columns = ['cell_type', 'count']

filtered_cells = cell_counts[(cell_counts['count'] >= 30)]
filtered_cells = filtered_cells.sort_values(by='count', ascending=False)

filtered_cells.to_csv("/gpfs/commons/home/nkeung/tabula_sapiens/cell_counts.tsv", sep="\t", index=False)
print(filtered_cells)
