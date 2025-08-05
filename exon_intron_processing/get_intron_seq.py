"""
Given the exon intron coordinate it finds the sequence of all species
"""

import pandas as pd
from pyfaidx import Fasta
import pickle
import os
import argparse
import time

start_totalcode = time.time()
## (AT)
IntronExonPosFile_default = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/knownGene.multiz100way.exonNuc_exon_intron_positions_exon1neg.csv' # Load the exon/intron positions CSV
parser = argparse.ArgumentParser(description="Process BAM files and output results.")
parser.add_argument("--IntronExonPosFile", type=str, default=IntronExonPosFile_default,
                    help=".csv file that stores the intron positions; default is /gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/dummy_exon_intron_positions.csv")

args = parser.parse_args()
csv_file_path = args.IntronExonPosFile
# species_url_csv = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/species_refSeq_urls.csv'     # Load the species-to-URL mapping from the CSV
species_url_csv = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/species_refSeq_urls_hg38.csv'     # (AT)
refseq_main_folder = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/refseq/'
output_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/initial_data/intron_no_dash/'

species_df = pd.read_csv(species_url_csv)
refseq_files = dict(zip(species_df['Species Name'], species_df['URL']))
df = pd.read_csv(csv_file_path)

# Initialize dictionary to store sequences for each exon
# exon_sequences_dict = {exon: [] for exon in df['Exon Name'].unique()}
exon_sequences_dict = {exon: {} for exon in df['Exon Name'].unique()}

# split_size = len(df) // 4
# filenames = ["split_file_1.csv", "split_file_2.csv", "split_file_3.csv", "split_file_4.csv"]
# for i, filename in enumerate(filenames):
#     start = i * split_size
#     end = (i + 1) * split_size if i < 3 else len(df)  # Make sure the last split includes any remaining rows
#     filename = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignmentknownGene.multiz100way.exonNuc_exon_intron_positions_'+filename
#     df[start:end].to_csv(filename, index=False)

# Define a helper function for reverse complement
def reverse_complement(seq):
    complement = str.maketrans("ATCGatcg", "TAGCtagc")
    return seq.translate(complement)[::-1]

total_introns = 0
ag_count = 0

# Function to get the intron sequence using pyfaidx
def get_intron_sequence(species, genome, chromosome, start, end, strand):
    global total_introns, ag_count  # Declare the counters as global so we can modify them here
    try:
        # Retrieve the sequence using pyfaidx (1-based indexing)
        intron_seq = genome[chromosome][int(start-1):int(end)].seq
        
        # Reverse complement if on the negative strand
        if strand == '-':
            intron_seq = reverse_complement(intron_seq)
        
        if intron_seq[-2:] == "AG":
            ag_count += 1
        total_introns += 1
        return intron_seq
    except Exception as e:
        # print(f"ERROR: Cannot retrieve intron species:{species} chrm {chromosome}:{start}-{end}: {e}") #(AT)
        return None

# Loop over each species and process exons
for species in refseq_files.keys():
    start = time.time()
    # Check if the species exists in the DataFrame
    if species not in df['Species Name'].values:
        print(f"ERROR: No MSA species {species}")
        continue

    print(f"Processing species: {species}")
    
    # Check for either .fa.gz or .fa files
    refseq_path_gz = refseq_main_folder + f'{species}.fa.gz'
    refseq_path_fa = refseq_main_folder + f'{species}.fa'
    
    # Determine the file to use
    if os.path.exists(refseq_path_gz):
        refseq_path = refseq_path_gz
    elif os.path.exists(refseq_path_fa):
        refseq_path = refseq_path_fa
    else:
        print(f"ERROR: No refseq species {species}")
        continue

    # Load the reference genome using pyfaidx
    try:
        genome = Fasta(refseq_path)
    except Exception as e:
        print(f"ERROR: Cannot load refseq species {species}: {e}") 
        continue

    # # Filter the dataframe to get only the rows for the current species
    species_df = df[df['Species Name'] == species]

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
            # Store the sequence with species name as the index
            exon_sequences_dict[exon_name][species] = intron_sequence

    # Apply the function to each row in the species-specific DataFrame
    # species_df.apply(extract_intron_sequence, axis=1) #(AT)
    positive_strand_df = species_df[species_df['Strand'] == '+']
    negative_strand_df = species_df[species_df['Strand'] == '-']
    
    print('postive strand')
    positive_strand_df.apply(extract_intron_sequence, axis=1)
    print(f"Positive strand: Total introns = {total_introns}, AG count = {ag_count}, Percentage = {(ag_count*100)/total_introns}")

    total_introns = 0
    ag_count = 0
    print('negative strand strand')
    negative_strand_df.apply(extract_intron_sequence, axis=1)
    print(f"Negative strand: Total introns = {total_introns}, AG count = {ag_count}, Percentage = {(ag_count*100)/total_introns}")



    end = time.time()
    print(f"time for species {species} {(end - start):.1f}s")



# Save the processed sequences as a pickle file or any other format you need
# output_name = output_path+(csv_file_path.split('/')[-1]).rsplit('.',1)[0]+'_IntronSeq.pkl'
# with open(output_name, 'wb') as f:
#     pickle.dump(exon_sequences_dict, f)

# print("Intron sequences have been successfully processed and saved.")
# end_totalcode = time.time()
# print(f"Total_time: {(end_totalcode-start_totalcode)/60}m")
