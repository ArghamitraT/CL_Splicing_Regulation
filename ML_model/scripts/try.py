import os
import pickle
import random

# === Configuration ===
pkl_file = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronSeq_multizAlignment_noDash/merged_intron_sequences.pkl"
output_dir = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronSeq_multizAlignment_noDash/"
# output_dir = os.path.dirname(pkl_file)

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# === Load full data ===
with open(pkl_file, "rb") as f:
    data = pickle.load(f)  # expected: dict[exon_name] -> stuff

exon_names = list(data.keys())
random.shuffle(exon_names)

n = len(exon_names)
n_train = int(train_ratio * n)
n_val = int(val_ratio * n)
n_test = n - n_train - n_val

train_keys = exon_names[:n_train]
val_keys   = exon_names[n_train:n_train + n_val]
test_keys  = exon_names[n_train + n_val:]

train_data = {k: data[k] for k in train_keys}
val_data   = {k: data[k] for k in val_keys}
test_data  = {k: data[k] for k in test_keys}

# === Save new .pkl files ===
with open(os.path.join(output_dir, "merged_intron_sequences_train.pkl"), "wb") as f:
    pickle.dump(train_data, f)
with open(os.path.join(output_dir, "merged_intron_sequences_val.pkl"), "wb") as f:
    pickle.dump(val_data, f)
with open(os.path.join(output_dir, "merged_intron_sequences_test.pkl"), "wb") as f:
    pickle.dump(test_data, f)

print(f"✅ Saved:")
print(f"  {len(train_data)} → train")
print(f"  {len(val_data)}   → val")
print(f"  {len(test_data)}  → test")
print(f"To folder: {output_dir}")
