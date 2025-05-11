import pickle

# Path to your pickle file
pkl_path = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/ASCOT_finetuning/psi_variable_Retina___Eye_psiWmean_intron_sequences_dict.pkl"

# Load the file
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

# Inspect the structure
print(type(data))        # Is it a dict, list, etc.?
print(len(data))         # How many entries?
print(next(iter(data)))  # Print the first key
print(data[next(iter(data))])  # Print the first value
