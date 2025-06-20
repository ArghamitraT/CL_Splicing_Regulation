"""
Extracts:
- 3' intron sequence
- 5' and 3' exon segments
for each exon/species using reference genomes.
"""


import pandas as pd
from pyfaidx import Fasta
import pickle
import os
import argparse
import time

start_totalcode = time.time()

# === Input arguments ===
# IntronExonPosFile_default = '/gpfs/commons/home/nkeung/data/knownGene.multiz100way.exonNuc_exon_intron_positions.csv'
IntronExonPosFile_default = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/exon_intron_positions_5and3prime/alignmentknownGene.multiz100way.exonNuc_exon_intron_positions_split_file_2.csv'
parser = argparse.ArgumentParser()
parser.add_argument("--IntronExonPosFile", type=str, default=IntronExonPosFile_default)
args = parser.parse_args()

csv_file_path = args.IntronExonPosFile
# species_url_csv = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/species_refSeq_urls_hg38.csv'
species_url_csv = '/gpfs/commons/home/nkeung/data/species_refSeq_urls.csv'
refseq_main_folder = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/refseq/'
output_path = '/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/ref_seq_processing'
os.makedirs(output_path, exist_ok=True)

# === Load inputs ===
species_df = pd.read_csv(species_url_csv)
refseq_files = dict(zip(species_df['Species Name'], species_df['URL']))
df = pd.read_csv(csv_file_path)

# === Initialize dicts ===
# intron_3prime_dict = {}
exon_parts_dict = {}

# === Helpers ===

def reverse_complement(seq):
    complement = str.maketrans("ACGT", "TGCA")
    return seq.translate(complement)[::-1]

def get_seq(genome, chrom, start, end, strand):
    try:
        seq = genome[chrom][int(start)-1:int(end)].seq
        return reverse_complement(seq) if strand == '-' else seq
    except:
        return None

def get_exon(genome, chrom, start, end, strand):
    try:
        exon_len = end - start + 1
        e_seq = genome[chrom][start-1:end].seq
        return reverse_complement(e_seq) if strand == '-' else e_seq
    except:
        return None, None

# === Main species loop ===
for species in refseq_files:
    if species not in df['Species_Name'].values:
        print(f"🚫 Skipping: {species}")
        continue

    print(f"🔍 Processing {species}")
    start = time.time()

    # Locate genome file
    fasta_gz = os.path.join(refseq_main_folder, f"{species}.fa.gz")
    fasta_raw = os.path.join(refseq_main_folder, f"{species}.fa")
    refseq_path = fasta_gz if os.path.exists(fasta_gz) else fasta_raw if os.path.exists(fasta_raw) else None

    if not refseq_path:
        print(f"🚫 Genome not found: {species}")
        continue

    try:
        genome = Fasta(refseq_path)
    except Exception as e:
        print(f"🚫 Failed to load genome for {species}: {e}")
        continue

    species_df = df[df['Species_Name'] == species]
    intron_count = 0
    exon_count = 0

    def extract(row):
        exon = row['Exon_Name']
        chrom = row['Chromosome']
        strand = row['Strand']

        # === 3' intron ===
        # intron = get_seq(genome, chrom, row['3prime_Intron_Start'], row['3prime_Intron_End'], strand)
        # if intron:
        #     intron_3prime_dict.setdefault(exon, {})[species] = intron

        # === exon segments ===
        ex_seq = get_exon(genome, chrom, row['Exon_Start'], row['Exon_End'], strand)
        if ex_seq:
            exon_parts_dict.setdefault(exon, {})        # format: dictionary[exon][species] = "ACTG..."
            exon_parts_dict[exon][species] = ex_seq

    # Strand-specific processing
    df_plus = species_df[species_df['Strand'] == '+']
    df_minus = species_df[species_df['Strand'] == '-']

    print("  ➕ Positive strand")
    df_plus.apply(extract, axis=1)

    print("  ➖ Negative strand")
    df_minus.apply(extract, axis=1)

    print(f"✅ {species} done in {(time.time() - start):.1f}s")

# === Save outputs ===
# intron_out = os.path.join(output_path, f"{base}_3primeIntronSeq.pkl")
exon_out = os.path.join(output_path, f"exon_nuc_seq.pkl")

# with open(intron_out, 'wb') as f:
#     pickle.dump(intron_3prime_dict, f)

with open(exon_out, 'wb') as f:
    pickle.dump(exon_parts_dict, f)

# print(f"\n✅ 3' intron saved to: {intron_out}")
print(f"✅ Exon start/end saved to: {exon_out}")
print(f"🕒 Total time: {(time.time() - start_totalcode)/60:.2f} min")
