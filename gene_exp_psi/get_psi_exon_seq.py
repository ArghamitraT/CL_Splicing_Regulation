"""
For each exons takes the intron coordinates from the csv files, gets the intron sequence
and saves in a pickle with psi values. 
"""

import pandas as pd
from pyfaidx import Fasta
import os
import time
import pickle
import numpy as np
from collections import Counter

# Load the intron positions CSV and the species-to-URL mapping
# csv_file_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/fine_tuning/Psi_values/types_Lung_with_psi_dummy.csv'  # Replace with your CSV file path

# csv_file_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/fine_tuning/Psi_values/types_Lung_psi.csv'  # Replace with your CSV file path
csv_file_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/fine_tuning/Psi_values/types_Lung_psi_exons_50to1000bp.csv'  # Replace with your CSV file path
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
    
max_exon_len = []

# Function to extract intron sequence and add to exon_sequences
def extract_intron_sequence(row):
    global max_exon_len
    exon_name = row['Exon Name']
    intron_sequence = get_intron_sequence(
        row['Species Name'],
        genome, 
        row['Chromosome'], 
        row['Exon Start'], 
        row['Exon End'], 
        row['Strand']
    )
    
    if intron_sequence:
        max_exon_len.append(len(intron_sequence)) 
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
    print(f'max_exon_len {max_exon_len}')
    print(f"Time for species {species}: {(end - start):.1f}s")


# Count occurrences of each length
length_counts = Counter(max_exon_len)

# Get the most common length(s)
most_common_length, count = length_counts.most_common(1)[0]

print(f"Most common length: {most_common_length} (appears {count} times)")

import matplotlib.pyplot as plt

# plt.hist(max_exon_len, bins=300, edgecolor='black')
# plt.title('Distribution of Intron Lengths')
# plt.xlabel('Length')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.xlim(0, 0.01e6)
# plt.savefig('/gpfs/commons/home/atalukder/Contrastive_Learning/code/figures/len_dis2.png')


# plt.hist(max_exon_len, bins=300, edgecolor='black', orientation='horizontal')
# plt.title('Distribution of Intron Lengths')
# plt.ylabel('Length')   # Now y-axis is length
# plt.xlabel('Frequency')  # Frequency is on x-axis
# plt.grid(True)
# plt.ylim(0, 0.01e6)  # Set y-axis limit since it's now flipped
# plt.savefig('/gpfs/commons/home/atalukder/Contrastive_Learning/code/figures/len_dis2.png')


lower = 50
upper = 1000

count_in_range = sum(lower <= x <= upper for x in max_exon_len)
percentage_in_range = (count_in_range / len(max_exon_len)) * 100

print(f"{percentage_in_range:.2f}% of exons fall between {lower} and {upper}")


lengths = np.array(max_exon_len)
sorted_lengths = np.sort(lengths)
cdf = np.arange(1, len(lengths)+1) / len(lengths)

import matplotlib.pyplot as plt

max_exon_len1= [x for x in max_exon_len if lower <= x <= upper]

plt.figure(figsize=(10, 6))
plt.hist(max_exon_len1, bins=300, density=True, edgecolor='black')  # `density=True` normalizes the histogram
plt.title('Normalized Distribution of Intron Lengths')
plt.xlabel('Intron Length')
plt.ylabel('Density (Fraction per unit length)')
plt.grid(True)
# plt.xlim(0, 8000)  # Focus on most common range
plt.tight_layout()
plt.savefig('/gpfs/commons/home/atalukder/Contrastive_Learning/code/figures/len_dis2.png')



arr = np.array(max_exon_len)



# Compute statistics
avg = np.mean(arr)
med = np.median(arr)
maximum = np.max(arr)
minimum = np.min(arr)
std_dev = np.std(arr)

# Print results
print(f"Average: {avg}")
print(f"Med: {med}")
print(f"Max: {maximum}")
print(f"Min: {minimum}")
print(f"Standard Deviation: {std_dev}")

# Save the processed sequences as a pickle file or any other format
# Extract the base file name without the path and extension
file_name = os.path.basename(csv_file_path)
tissue_type = file_name.split('_')[1]
output_name = os.path.join(output_path, f'psi_{tissue_type}_exon_sequences_dict_50to1000bp.pkl')
with open(output_name, 'wb') as f:
    pickle.dump(exon_sequences_dict, f)

print("Exon sequences have been successfully processed and saved.")

