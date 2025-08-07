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

pickle_file_path = '/mnt/home/at3836/Contrastive_Learning/data/final_data/ASCOT_finetuning/psi_test_Retina___Eye_psi_MERGED.pkl'
# pickle_file_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/test_5primeIntron_filtered.pkl'
with open(pickle_file_path, 'rb') as f:
    msa_results_list_unTOKEN = pickle.load(f)

print()