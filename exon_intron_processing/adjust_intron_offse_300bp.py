import pandas as pd

# === Input and output paths ===
main_dir = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/exon_intron_positions_5and3prime/'
division = '4'
input_file = f'{main_dir}alignmentknownGene.multiz100way.exonNuc_exon_intron_positions_split_file_{division}.csv'
output_file = f'{main_dir}alignmentknownGene.multiz100way.exonNuc_exon_intron_positions_split_file_{division}_adjustedIntron300bpOffset.csv'

# === Load the CSV ===
df = pd.read_csv(input_file)
print(f"Loaded {len(df)} rows")

# === Convert columns to numeric for safety ===
for col in [
    "5prime_Intron_Start", "5prime_Intron_End",
    "3prime_Intron_Start", "3prime_Intron_End"
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# === Compute current intron lengths ===
df["len_5prime"] = df["5prime_Intron_End"] - df["5prime_Intron_Start"]
df["len_3prime"] = df["3prime_Intron_End"] - df["3prime_Intron_Start"]

print("Before adjustment:")
print(df[["len_5prime", "len_3prime"]].describe())

# === Adjust to have 300 bp offset ===
# Keep start same, move end = start + 300
df["5prime_Intron_End"] = df["5prime_Intron_Start"] + 300
df["3prime_Intron_End"] = df["3prime_Intron_Start"] + 300

# === Save updated file ===
df.drop(columns=["len_5prime", "len_3prime"]).to_csv(output_file, index=False)
print(f"✅ Saved updated file: {output_file}")
