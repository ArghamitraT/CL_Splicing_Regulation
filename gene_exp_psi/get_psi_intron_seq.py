"""
For each exons takes the intron coordinates from the csv files, gets the intron sequence
and saves in a pickle with psi values. 
"""

import pandas as pd
from pyfaidx import Fasta
import os
import time
import pickle

# Load the intron positions CSV and the species-to-URL mapping
# csv_file_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/fine_tuning/Psi_values/types_Lung_with_psi_dummy.csv'  # Replace with your CSV file path

csv_file_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/fine_tuning/Psi_values/types_Lung_psi.csv'  # Replace with your CSV file path
refseq_main_folder = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/refseq/'  # Replace with your reference sequence folder
output_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/fine_tuning/Psi_values/'  # Replace with your desired output path

df = pd.read_csv(csv_file_path)
exon_sequences_dict = {exon: {'psi_val': None} for exon in df['Exon Name'].unique()}

# Initialize counters for total introns and AG-ending introns
total_introns_pos, total_introns_neg = 0, 0
ag_count_pos, ag_count_neg = 0, 0

# Function to calculate reverse complement
def reverse_complement(seq):
    complement = str.maketrans("ATCGatcg", "TAGCtagc")
    return seq.translate(complement)[::-1]

# Function to get the intron sequence and count "AG" endings
def get_intron_sequence(species, genome, chromosome, start, end, strand):
    global total_introns_pos, total_introns_neg, ag_count_pos, ag_count_neg  # Global counters
    try:
        # Retrieve the sequence (1-based indexing)
        intron_seq = genome[chromosome][int(start)-1:int(end)].seq
        
        # Reverse complement if on the negative strand
        if strand == '-':
            intron_seq = reverse_complement(intron_seq)
        
        # Check for "AG" ending and increment the appropriate counters
        if strand == '+':
            total_introns_pos += 1
            if intron_seq[-2:] == "AG":
                ag_count_pos += 1
        else:
            total_introns_neg += 1
            if intron_seq[-2:] == "AG":
                ag_count_neg += 1

        return intron_seq
    except Exception as e:
        print(f"ERROR: Cannot retrieve intron for {species} {chromosome}:{start}-{end}: {e}")
        return None

# Function to extract intron sequence and add to exon_sequences
def extract_intron_sequence(row):
    exon_name = row['Exon Name']
    intron_sequence = get_intron_sequence(
        row['Species Name'],
        genome, 
        row['Chromosome'], 
        row['Intron Start'], 
        row['Intron End'], 
        row['Strand']
    )
    
    if intron_sequence:
        exon_sequences_dict[exon_name][species] = intron_sequence
        if exon_sequences_dict[exon_name]['psi_val'] is None:
            exon_sequences_dict[exon_name]['psi_val'] = row['PSI']


species_arr = ['hg38']
# # Loop over each species and process exons
for species in species_arr:
    start = time.time()
    if species not in df['Species Name'].values:
        continue

    print(f"Processing species: {species}")
    
    # Determine the path to the reference genome
    refseq_path_gz = os.path.join(refseq_main_folder, f'{species}.fa.gz')
    refseq_path_fa = os.path.join(refseq_main_folder, f'{species}.fa')
    
    if os.path.exists(refseq_path_gz):
        refseq_path = refseq_path_gz
    elif os.path.exists(refseq_path_fa):
        refseq_path = refseq_path_fa
    else:
        print(f"ERROR: No refseq file for {species}")
        continue

    try:
        genome = Fasta(refseq_path)
    except Exception as e:
        print(f"ERROR: Cannot load refseq for {species}: {e}") 
        continue

    # Filter for species-specific data and separate by strand
    species_df = df[df['Species Name'] == species]
    positive_strand_df = species_df[species_df['Strand'] == '+']
    negative_strand_df = species_df[species_df['Strand'] == '-']

    # Process positive strand
    print("Processing positive strand")
    total_introns_pos, ag_count_pos = 0, 0  # Reset counters
    positive_strand_df.apply(extract_intron_sequence, axis=1)
    print(f"Positive strand: Total introns = {total_introns_pos}, AG count = {ag_count_pos}, Percentage = {(ag_count_pos * 100) / total_introns_pos if total_introns_pos > 0 else 0:.2f}%")

    # Process negative strand
    print("Processing negative strand")
    total_introns_neg, ag_count_neg = 0, 0  # Reset counters
    negative_strand_df.apply(extract_intron_sequence, axis=1)
    print(f"Negative strand: Total introns = {total_introns_neg}, AG count = {ag_count_neg}, Percentage = {(ag_count_neg * 100) / total_introns_neg if total_introns_neg > 0 else 0:.2f}%")

    end = time.time()
    print(f"Time for species {species}: {(end - start):.1f}s")

# Save the processed sequences as a pickle file or any other format
# Extract the base file name without the path and extension
file_name = os.path.basename(csv_file_path)
tissue_type = file_name.split('_')[1]
output_name = os.path.join(output_path, f'psi_{tissue_type}_intron_sequences_dict.pkl')
with open(output_name, 'wb') as f:
    pickle.dump(exon_sequences_dict, f)

print("Intron sequences have been successfully processed and saved.")






# import pandas as pd
# from pyfaidx import Fasta
# import os
# import time
# import pickle

# # Load the intron positions CSV and the species-to-URL mapping
# csv_file_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/fine_tuning/ConstitutiveVSAlternative/Constitutives_exons_with_introns.csv'  # Replace with your CSV file path
# # species_url_csv = '/path/to/species_refSeq_urls_hg38.csv'  # Replace with your species-URL CSV
# refseq_main_folder = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/refseq/'  # Replace with your reference sequence folder
# output_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/Constitutive_fineTuning/'  # Replace with your desired output path

# # species_df = pd.read_csv(species_url_csv)
# # refseq_files = dict(zip(species_df['Species Name'], species_df['URL']))
# df = pd.read_csv(csv_file_path)

# # Initialize dictionary to store sequences for each exon
# # exon_sequences_dict = {exon: {'hg38': {}, 'constitutive': None} for exon in df['Exon Name'].unique()}
# exon_sequences_dict = {exon: {'constitutive': None} for exon in df['Exon Name'].unique()}

# # Define a helper function for reverse complement
# def reverse_complement(seq):
#     complement = str.maketrans("ATCGatcg", "TAGCtagc")
#     return seq.translate(complement)[::-1]

# # Function to get the intron sequence using pyfaidx
# def get_intron_sequence(species, genome, chromosome, start, end, strand):
#     try:
#         # Retrieve the sequence using pyfaidx (1-based indexing)
#         intron_seq = genome[chromosome][int(start)-1:int(end)].seq
        
#         # Reverse complement if on the negative strand
#         if strand == '-':
#             intron_seq = reverse_complement(intron_seq)
        
#         return intron_seq
#     except Exception as e:
#         print(f"ERROR: Cannot retrieve intron for species:{species} chrm {chromosome}:{start}-{end}: {e}")
#         return None

# species_arr = ['hg38']
# # Loop over each species and process exons
# for species in species_arr:
#     start = time.time()
    
#     # Skip if the species doesn't exist in the DataFrame
#     if species not in df['Species Name'].values:
#         print(f"ERROR: No MSA species {species}")
#         continue

#     print(f"Processing species: {species}")
    
#     # Determine the file path for the reference genome
#     refseq_path_gz = os.path.join(refseq_main_folder, f'{species}.fa.gz')
#     refseq_path_fa = os.path.join(refseq_main_folder, f'{species}.fa')
    
#     # Check if the reference file exists
#     if os.path.exists(refseq_path_gz):
#         refseq_path = refseq_path_gz
#     elif os.path.exists(refseq_path_fa):
#         refseq_path = refseq_path_fa
#     else:
#         print(f"ERROR: No refseq file for species {species}")
#         continue

#     # Load the reference genome using pyfaidx
#     try:
#         genome = Fasta(refseq_path)
#     except Exception as e:
#         print(f"ERROR: Cannot load refseq for species {species}: {e}") 
#         continue

#     # Filter the dataframe to get only the rows for the current species
#     species_df = df[df['Species Name'] == species]
    

#     # Loop over each exon in the species-specific DataFrame and extract intron sequences
#     for _, row in species_df.iterrows():
#         exon_name = row['Exon Name']
#         intron_sequence = get_intron_sequence(
#             row['Species Name'],
#             genome, 
#             row['Chromosome'], 
#             row['Intron Start'], 
#             row['Intron End'], 
#             row['Strand']
#         )
        
#         # Store intron sequence if retrieved successfully
#         if intron_sequence:
#             exon_sequences_dict[exon_name][species] = intron_sequence
#             # Store the constitutive status (assumed consistent across species for each exon)
#             if exon_sequences_dict[exon_name]['constitutive'] is None:
#                 exon_sequences_dict[exon_name]['constitutive'] = row['Constitutive Exon']

#     end = time.time()
#     print(f"Time for species {species}: {(end - start):.1f}s")

# # Save the processed sequences as a pickle file or any other format you need
# output_name = os.path.join(output_path, 'Constitutive_intron_sequences_dict.pkl')
# with open(output_name, 'wb') as f:
#     pickle.dump(exon_sequences_dict, f)

# print("Intron sequences have been successfully processed and saved.")
