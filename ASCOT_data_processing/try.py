import pandas as pd

# # Load the CSV file
# input_csv = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/dummy_exon_intron_positions_shrt.csv'  # Replace with your file path
# output_bed = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/dummy_exon_intron_positions_shrt.bed'  # Output BED file name

# # Read the CSV file
# df = pd.read_csv(input_csv)

# # Select the required columns and add a placeholder column
# #  Columns required are: chromosome, start, end, name (4 fields)
# bed_df = df[['Chromosome', 'Exon Start', 'Exon End']]

# # Ensure Exon Start and Exon End are integers
# bed_df['Exon Start'] = bed_df['Exon Start'].astype(int)
# bed_df['Exon End'] = bed_df['Exon End'].astype(int)

# # Remove duplicates again to be thorough
# bed_df = bed_df.drop_duplicates(subset=['Chromosome', 'Exon Start', 'Exon End'])

# # Add a unique identifier as the fourth column to avoid duplication issues
# bed_df['UniqueID'] = range(1, len(bed_df) + 1)

# # Save as a BED file with tab-separated values
# bed_df.to_csv(output_bed, sep='\t', header=False, index=False)

# print(f"BED file saved as {output_bed}")



# import pandas as pd
# from gpn.data import GenomeMSA
import pickle

# # pickle_file_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/initial_data/msa_results_TOKEN.pkl'
# # with open(pickle_file_path, 'rb') as f:
# #     msa_results_list_TOKEN = pickle.load(f)

# pickle_file_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/ASCOT_data/train_ExonExon_meanAbsDist_ASCOTname.pkl'
# # pickle_file_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/test_5primeIntron_filtered.pkl'
# with open(pickle_file_path, 'rb') as f:
#     msa_results_list_unTOKEN = pickle.load(f)

# print(pickle_file_path)
# print(len(msa_results_list_unTOKEN))


# --- 1. Load your pickle file ---
import pickle
import numpy as np
import pandas as pd

pickle_file_path = '/mnt/home/at3836/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/ASCOT_data/train_ExonExon_meanAbsDist_ASCOTname.pkl'
with open(pickle_file_path, 'rb') as f:
    data = pickle.load(f)

print(f"File loaded: {pickle_file_path}")
print(f"Data type: {type(data)}")


#  --- 2. Check for NaNs based on the object type ---

if isinstance(data, np.ndarray):
    print("Data is a NumPy array. Checking for NaNs...")
    # Check if any NaNs exist at all
    has_nan = np.isnan(data).any()
    
    if has_nan:
        print("🔴 NaN values found!")
        # Find the row and column indices of all NaNs
        nan_rows, nan_cols = np.where(np.isnan(data))
        print(f"Found {len(nan_rows)} NaN(s) at the following (row, column) coordinates:")
        # Print the coordinates
        for r, c in zip(nan_rows, nan_cols):
            print(f"  - (Row: {r}, Column: {c})")
    else:
        print("✅ No NaN values found in the array.")

elif isinstance(data, pd.DataFrame):
    print("Data is a pandas DataFrame. Checking for NaNs...")
    # Check if any NaNs exist at all
    has_nan = data.isna().any().any()

    if has_nan:
        print("🔴 NaN values found!")
        # Find the row and column indices of all NaNs
        nan_rows, nan_cols = np.where(data.isna())
        print(f"Found {len(nan_rows)} NaN(s) at the following (row, column) coordinates:")
        for r, c in zip(nan_rows, nan_cols):
            # You can get labels if you need them: data.index[r], data.columns[c]
            print(f"  - (Row index: {r}, Column index: {c})")
    else:
        print("✅ No NaN values found in the DataFrame.")
        
else:
    print("Data is not a NumPy array or pandas DataFrame. Cannot perform NaN check.")
