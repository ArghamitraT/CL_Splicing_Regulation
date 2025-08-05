import torch

em_file = "/gpfs/commons/home/nkeung/data/embeddings/hg38_exons.pt"

flattened = torch.load(em_file)
print(f"Number of exons: {len(flattened)}")

nested_emb = {}
for (gene, exon), vector in flattened.items():
    nested_emb.setdefault(gene, {})[exon] = vector

print(f"Number of genes: {len(nested_emb)}")
