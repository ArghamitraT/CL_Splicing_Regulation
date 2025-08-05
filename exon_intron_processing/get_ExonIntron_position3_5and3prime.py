"""
Given the exon coordinated csv file, it adds column 3 prime of intron coordinates and exon length
"""



import pandas as pd
import os
import numpy as np

# === Input and output file paths ===
input_file = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/exon_intron_positions_only5prime/alignmentknownGene.multiz100way.exonNuc_exon_intron_positions_split_file_4.csv'
output_file = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/exon_intron_positions_5and3prime/alignmentknownGene.multiz100way.exonNuc_exon_intron_positions_split_file_4.csv'

# === Make sure output directory exists ===
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# === Read CSV ===
df = pd.read_csv(input_file)

# === Rename 5′ intron columns ===
df.rename(columns={
    'Intron Start': '5prime Intron Start',
    'Intron End': '5prime Intron End'
}, inplace=True)

# # === Compute 3′ intron coordinates using vectorized NumPy operations ===
# plus_strand = df['Strand'] == '+'
# minus_strand = ~plus_strand
# start_gt_200 = df['Exon Start'] > 200

# df['3prime Intron Start'] = np.where(
#     plus_strand,
#     df['Exon End'] + 1,
#     np.where(start_gt_200, df['Exon Start'] - 200, np.nan)
# )

# df['3prime Intron End'] = np.where(
#     plus_strand,
#     df['Exon End'] + 200,
#     np.where(start_gt_200, df['Exon Start'] - 1, np.nan)
# )


# Create masks
plus_strand = df['Strand'] == '+'
minus_strand = df['Strand'] == '-'
start_gt_200 = df['Exon Start'] > 200

# Initialize with NaNs
df['3prime Intron Start'] = np.nan
df['3prime Intron End'] = np.nan

# For + strand: exon_end + [1, 200]
df.loc[plus_strand, '3prime Intron Start'] = df.loc[plus_strand, 'Exon End'] + 1
df.loc[plus_strand, '3prime Intron End']   = df.loc[plus_strand, 'Exon End'] + 200

# For - strand: exon_start - [200, 1] — only if exon_start > 200
valid_minus = minus_strand & (df['Exon Start'] > 200)
df.loc[valid_minus, '3prime Intron Start'] = df.loc[valid_minus, 'Exon Start'] - 200
df.loc[valid_minus, '3prime Intron End']   = df.loc[valid_minus, 'Exon Start'] - 1


# === Add Exon Length if not present ===
if 'Exon Length' not in df.columns:
    df['Exon Length'] = df['Exon End'] - df['Exon Start'] + 1

# === Replace spaces with underscores in all column names ===
df.columns = [col.replace(' ', '_') for col in df.columns]

# === Save updated DataFrame ===
df.to_csv(output_file, index=False)
print(f"✅ Saved updated file: {output_file}")



# import pandas as pd
# import os
# from glob import glob

# # === Input/output paths ===
# csv_dir = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/exon_intron_positions_only5prime/alignmentknownGene.multiz100way.exonNuc_exon_intron_positions_split_file_1.csv'
# output_dir = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/exon_intron_positions_5and3prime/alignmentknownGene.multiz100way.exonNuc_exon_intron_positions_split_file_1.csv'
# os.makedirs(output_dir, exist_ok=True)

# # Process each CSV file
# # for file_path in glob(os.path.join(csv_dir, '*.csv')):
# #     df = pd.read_csv(file_path)

# # for file_path in glob(os.path.join(csv_dir, '*.csv')):
# df = pd.read_csv(csv_dir)

# # Rename 5′ intron columns
# df.rename(columns={
#     'Intron Start': '5prime Intron Start',
#     'Intron End': '5prime Intron End'
# }, inplace=True)

# # Add 3′ intron columns
# def get_3prime(row):
#     if row['Strand'] == '+':
#         start = row['Exon End'] + 1
#         end = row['Exon End'] + 200
#     else:
#         if row['Exon Start'] > 200:
#             start = row['Exon Start'] - 200
#             end = row['Exon Start'] - 1
#         else:
#             start, end = None, None
#     return pd.Series([start, end])

# df[['3prime Intron Start', '3prime Intron End']] = df.apply(get_3prime, axis=1)

# # Add exon length if not present
# if 'Exon Length' not in df.columns:
#     df['Exon Length'] = df['Exon End'] - df['Exon Start'] + 1

# # Replace spaces with underscores in column names
# df.columns = [col.replace(' ', '_') for col in df.columns]

# # Save to new file
# # out_file = os.path.join(output_dir, os.path.basename(file_path).replace('only5prime', '5and3prime'))
# out_file = output_dir
# df.to_csv(out_file, index=False)
# print(f"✅ Saved updated file: {out_file}")
