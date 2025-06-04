import os
import pickle
import random

def get_exon_keys(file_list, data_dir):
    exon_keys = set()
    for fname in file_list:
        path = os.path.join(data_dir, fname)
        with open(path, 'rb') as f:
            data = pickle.load(f)
            exon_keys.update(data.keys())
    return exon_keys

def split_exons(all_exons, seed=42, val_ratio=0.1, test_ratio=0.1):
    exons = list(all_exons)
    random.seed(seed)
    random.shuffle(exons)
    n = len(exons)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    val = exons[:n_val]
    test = exons[n_val:n_val + n_test]
    train = exons[n_val + n_test:]
    return train, val, test

def merge_and_filter(file_list, data_dir, target_exons):
    filtered = {}
    for fname in file_list:
        path = os.path.join(data_dir, fname)
        with open(path, 'rb') as f:
            data = pickle.load(f)
            for exon in target_exons:
                if exon in data:
                    filtered[exon] = data[exon]
    return filtered

def save_pickle(data, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

def main():
    data_dir = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/unmerged_pkl_3prime5primeIntronExon/"

    data_types = {
        "3primeIntron": [f"alignmentknownGene.multiz100way.exonNuc_exon_intron_positions_split_file_{i}_3primeIntronSeq.pkl" for i in range(1, 5)],
        "5primeIntron": [f"alignmentknownGene.multiz100way.exonNuc_exon_intron_positions_split_file_{i}_5primeIntronSeq.pkl" for i in range(1, 5)],
        "ExonSeq":      [f"alignmentknownGene.multiz100way.exonNuc_exon_intron_positions_split_file_{i}_ExonSeq.pkl" for i in range(1, 5)],
    }

    print("🔍 Scanning keys to compute shared exons...")
    key_sets = []
    for dtype, files in data_types.items():
        keys = get_exon_keys(files, data_dir)
        print(f"✅ Found {len(keys)} exons in {dtype}")
        key_sets.append(keys)

    shared_exons = set.intersection(*key_sets)
    print(f"📌 Shared exons across all 3 types: {len(shared_exons)}")

    train, val, test = split_exons(shared_exons)

    # One data type at a time (to save memory)
    for dtype, files in data_types.items():
        print(f"\n💾 Processing {dtype}...")

        split_map = {
            "train": train,
            "val": val,
            "test": test
        }

        for split_name, exon_list in split_map.items():
            print(f"   - Filtering {split_name} set with {len(exon_list)} exons...")
            filtered = merge_and_filter(files, data_dir, exon_list)
            out_path = os.path.join(data_dir, f"{split_name}_{dtype}.pkl")
            save_pickle(filtered, out_path)
            print(f"   ✅ Saved {split_name}_{dtype}.pkl with {len(filtered)} entries")

if __name__ == "__main__":
    main()
