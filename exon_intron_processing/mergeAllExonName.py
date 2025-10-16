import pandas as pd

# 1. Define the paths to your three exon list files
# (Replace with your actual file names)
main_dir = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/all_names"
exon_file_paths = {
    'train': f'{main_dir}/train_exon_list.csv',
    'val': f'{main_dir}/val_exon_list.csv',
    'test': f'{main_dir}/test_exon_list.csv'
}

# 2. Load and combine all exon IDs
all_exons_list = []
for split, path in exon_file_paths.items():
    df = pd.read_csv(path)
    # Assuming the column is named 'exon_id' as in your image
    all_exons_list.append(df['exon_id'])

# 3. Concatenate into a single series and get unique values
combined_exons = pd.concat(all_exons_list)
master_exon_df = combined_exons.drop_duplicates().reset_index(drop=True).to_frame(name='exon_id')

# 4. Save the final master list
master_exon_df.to_csv(f'{main_dir}/all_exon_list.csv', index=False)

print(f"Created a master list with {len(master_exon_df)} unique exon names.")
