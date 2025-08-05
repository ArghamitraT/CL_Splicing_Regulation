import pandas as pd

main_dir = '/home/argha/Contrastive_Learning/data/ASCOT/'
# Load full dataset
df = pd.read_csv(main_dir+"gtex_psi.csv")

# Only keep cassette exons
cassette_df = df[df["cassette_exon"] == "Yes"].copy()

# Extract chromosome name from exon_location (e.g., "chr1:123-456" → "chr1")
cassette_df["chromosome"] = cassette_df["exon_location"].str.extract(r'(chr[0-9XY]+)')

# Define chromosome groups
train_chroms = [f'chr{i}' for i in range(10, 23)] + ['chr4', 'chr6', 'chr8', 'chrX', 'chrY']
val_chroms = ['chr1', 'chr7', 'chr9']
test_chroms = ['chr2', 'chr3', 'chr5']

# Create masks
train_mask = cassette_df["chromosome"].isin(train_chroms)
val_mask = cassette_df["chromosome"].isin(val_chroms)
test_mask = cassette_df["chromosome"].isin(test_chroms)

# Apply splits
train_df = cassette_df[train_mask]
val_df = cassette_df[val_mask]
test_df = cassette_df[test_mask]

# Save splits
train_df.to_csv(main_dir+"train_cassette_exons.csv", index=False)
val_df.to_csv(main_dir+"val_cassette_exons.csv", index=False)
test_df.to_csv(main_dir+"test_cassette_exons.csv", index=False)
