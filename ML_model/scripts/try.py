


import pickle
import random
import os

# === Paths ===
main_dir = "/mnt/home/at3836/Contrastive_Learning/data/final_data/ASCOT_finetuning"
file = "psi_val_Retina___Eye_psi_MERGED.pkl"
path = f"{main_dir}/{file}"

# === Output directory ===
out_dir = "/mnt/home/at3836/Contrastive_Learning/RECOMB_26/CLADES/data/finetune_sample_data"
os.makedirs(out_dir, exist_ok=True)

out_file = f"{out_dir}/{file}"

# === Load original data ===
with open(path, "rb") as f:
    data = pickle.load(f)

# print("Total exons:", len(data))

# === Sample 1% of exon IDs ===
keys = list(data.keys())
n = max(1, int(len(keys) * 0.01))  # at least 1
sampled_keys = random.sample(keys, n)

# === Create new dictionary ===
sampled_data = {k: data[k] for k in sampled_keys}

# print("Sampled exons:", len(sampled_data))

# === Save ===
with open(out_file, "wb") as f:
    pickle.dump(sampled_data, f)

print(f"Saved 1% sample to: {out_file}")
