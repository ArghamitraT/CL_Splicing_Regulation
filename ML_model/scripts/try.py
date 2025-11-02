import pickle

file = 'filtered_HiLow_exons_by_tissue'
path = f"/gpfs/commons/home/atalukder/Contrastive_Learning/data/extra/{file}.pkl"
with open(path, "rb") as f:
    data = pickle.load(f)

print(data)