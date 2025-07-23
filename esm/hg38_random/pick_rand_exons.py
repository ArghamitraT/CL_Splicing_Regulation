import json
import pandas as pd

all_exons = "/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/hg38_all_exons.csv"
output_pairs = "/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/rand_pairs.json"
stitched_genes = "/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/hg38_rand_seqs.json"

needed_genes = set()
pairs = []

df = pd.read_csv(all_exons)

print(f"Sampling random genes (2000 pairs)...")
count = 0
while count < 2000:
    pair_df = df.sample(2).reset_index(drop=True)
    
    gene1, exon1 = pair_df.loc[0, "Gene"], pair_df.loc[0, "Number"]
    gene2, exon2 = pair_df.loc[1, "Gene"], pair_df.loc[1, "Number"]
    if gene1 == gene2:
        continue
    needed_genes.update([gene1, gene2])
    pairs.append( ((gene1, int(exon1)), (gene2, int(exon2))) )
    count += 1

needed_genes = frozenset(needed_genes)
print(f"Number of genes needed: {len(needed_genes)}")
with open(output_pairs, "w") as f:
    json.dump(pairs, f)
print(f"✅ Saved random exon pairs in {output_pairs}\n")


print(f"Stitching genes together...\n")
aa_seqs = []

filtered_df = df[df["Gene"].isin(needed_genes)]
filtered_df = filtered_df.sort_values(by=["Gene", "Number"])
grouped_df = filtered_df.groupby("Gene")

for gene, groups in grouped_df:
    sequence = []
    for exon in groups["Seq"]:
        if "Z" in exon:
            sequence.append(exon.split("Z",1)[0])
            break
        else:
            sequence.append(exon)
    if sequence:
        str_seq = "".join(sequence).replace("-", "").replace("\n", "")
        aa_seqs.append((gene, str_seq))

with open(stitched_genes, "w") as f:
    json.dump(aa_seqs, f)

print(f"Saved {len(aa_seqs)} sequences.")
print(f"✅ Saved all sequences as {stitched_genes}")