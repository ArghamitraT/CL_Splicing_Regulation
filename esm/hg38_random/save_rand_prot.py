import pickle
import json
import random
import pandas

exon_seqs = f"/gpfs/commons/home/nkeung/data/hg38_knownGene.exonAA.fa"

with open(f"/gpfs/commons/home/nkeung/data/human_exon_indices.pkl", "rb") as f:
    indices = pickle.load(f)
proteins = list(indices.keys())

sequences = []
with open(exon_seqs, "r") as f:
    for i in range(1000):
        rand_prot = random.choice(proteins)
        seq = ""
        for i in range(1, 501):
            try:
                f.seek(indices[rand_prot][str(i)])
                header = f.readline()
                if rand_prot not in header:
                    print(f"Error searching for {rand_prot}, exon {i}")
                    continue
                aa = f.readline().strip()
                seq += aa
            except KeyError:
                # Found last exon
                break
        if "Z" in seq:
            seq = seq[:seq.index("Z")]
        sequences.append((rand_prot, seq))

with open("/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/hg38-rand-seqs.json", "w") as f:
    json.dump(sequences, f)
