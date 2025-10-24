import pandas as pd
import os
import numpy as np

def get_5prime_and_3prime_coords(exon_start, exon_end, strand):
    """
    Given exon coordinates and strand, return (5′_start, 5′_end, 3′_start, 3′_end).
    """
    if strand == '+':
        intron_5p_start = exon_start - 300 if exon_start > 300 else None
        intron_5p_end = exon_start - 1 if exon_start > 300 else None
        intron_3p_start = exon_end + 1
        intron_3p_end = exon_end + 300
    elif strand == '-' and exon_end > 300:
        intron_5p_start = exon_end + 1
        intron_5p_end = exon_end + 300
        intron_3p_start = exon_start - 300 if exon_start > 300 else None
        intron_3p_end = exon_start - 1 if exon_start > 300 else None
    else:
        intron_5p_start = intron_5p_end = intron_3p_start = intron_3p_end = None

    return intron_5p_start, intron_5p_end, intron_3p_start, intron_3p_end


# === ✅ Test Block ===
def reverse_complement(seq):
    complement = str.maketrans("ACGT", "TGCA")
    return seq.translate(complement)[::-1]

test_examples = [
    {"exon_start": 300, "exon_end": 305, "strand": "+", "seq": "ATCGGA"},
    {"exon_start": 300, "exon_end": 305, "strand": "-", "seq": "ATCGGA"},
]

print("==== Test Cases ====")
for e in test_examples:
    intron_5p_s, intron_5p_e, intron_3p_s, intron_3p_e = get_5prime_and_3prime_coords(e["exon_start"], e["exon_end"], e["strand"])
    exon_seq = e["seq"] if e["strand"] == "+" else reverse_complement(e["seq"])
    print(f"Strand: {e['strand']}")
    print(f"Exon: {e['exon_start']}–{e['exon_end']}, Exon Seq (5′→3′): {exon_seq}")
    print(f"5′ Intron: {intron_5p_s}–{intron_5p_e}")
    print(f"3′ Intron: {intron_3p_s}–{intron_3p_e}\n")


# === ✅ Main Processing Script ===

# Input and output file paths
input_file = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/knownGene.multiz100way.exonNuc_exon_intron_positions.csv'
# input_file = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/dummy_exon_intron_positions_shrt.csv'
output_file = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/knownGene.multiz100way.exonNuc_exon_intron_positions_computedIntron300bpOffset.csv'
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# === Read CSV and drop original intron columns ===
df = pd.read_csv(input_file)
df = df.drop(columns=["Intron Start", "Intron End"], errors="ignore")

# === Compute new intron coordinates ===
df[['5prime Intron Start', '5prime Intron End', '3prime Intron Start', '3prime Intron End']] = df.apply(
    lambda row: pd.Series(get_5prime_and_3prime_coords(row['Exon Start'], row['Exon End'], row['Strand'])),
    axis=1
)

# Add Exon Length
df['Exon Length'] = df['Exon End'] - df['Exon Start'] + 1

# Replace spaces with underscores in column names
df.columns = [col.replace(' ', '_') for col in df.columns]


# === Save updated DataFrame ===
df.to_csv(output_file, index=False)
print(f"✅ Saved updated file: {output_file}")