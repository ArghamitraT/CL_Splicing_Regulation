


import pickle

file = '/home/atalukder/Contrastive_Learning/data/final_data/ASCOT_finetuning/psi_variable_Retina___Eye_psi_MERGED.pkl'
# Replace with your actual file path
with open(file, "rb") as f:
    data = pickle.load(f)

print(type(data))
print(data)
