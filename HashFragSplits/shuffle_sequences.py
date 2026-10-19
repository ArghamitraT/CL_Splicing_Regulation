
import os
import random
import ushuffle
from Bio import SeqIO
from Bio.Seq import Seq

random.seed(42)

DIR = "/gpfs/commons/home/nkeung/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/chromSplit/hashFragData"

splits = ["val", "test", "train"]
shuffled_records = []


for split in splits:
    input_file = os.path.join(DIR, f"{split}_hg38_sequences.fa")

    seqs = list(SeqIO.parse(input_file, "fasta"))
    n = 2000 if split in ("val", "test") else 8000
    sampled_seqs = random.sample(seqs, min(n, len(seqs)))

    for record in sampled_seqs:
        seq = str(record.seq).upper()
        shuffled = ushuffle.shuffle(seq, len(seq), 2)

        shuffled_record = record[:]
        shuffled_record.id = record.id + "_shuffled"
        shuffled_record.description = ""
        shuffled_record.seq = Seq(shuffled)
        shuffled_records.append(shuffled_record)

output_file = os.path.join(DIR, "dinucleotide_shuffled_hg38.fa")
SeqIO.write(shuffled_records, output_file, "fasta-2line")

print(f"Wrote {len(shuffled_records)} shuffled sequences to {output_file}")
