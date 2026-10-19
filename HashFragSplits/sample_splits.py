import os
from random import sample
from Bio import SeqIO

DIR = "/gpfs/commons/home/nkeung/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/chromSplit/hashFragData"

splits = ["val", "test", "train"]

for split in splits:
    print(f"Sampling {split}:")
    input_file = os.path.join(DIR, f"{split}_hg38_sequences.fa")

    with open(input_file, "r") as f:
        seqs = list(SeqIO.parse(f, "fasta"))

    if split == "val" or split == "test":
        n = 2000
    else:
        n = 8000

    sampled_seqs = sample(seqs, n)
    output_file = os.path.join(DIR, f"sample_{split}_hg38_sequences.fa")
    SeqIO.write(sampled_seqs, output_file, "fasta-2line")
    print(f"\t{split} done")

print("DONE")
