"""
Searches for a given gene and its amino acid alignment with other species.
"""

import pandas as pd
# import gzip
import re
import pickle

alignment_file = "knownGene.exonAA.fa"
# alignment_file = "test_truncated.fa"
input_file = "/gpfs/commons/home/nkeung/data/" + alignment_file

enst_id = {"foxp2": "ENST00000350908.9", "brca2": "ENST00000380152.7", "hla-a": "ENST00000376809.10"}
gene = "brca2"
# enst_id = "ENST00000350908.9"     # foxp2
enst_id = "ENST00000380152.7"     # brca2
# enst_id = "ENST00000376809.10"      # hla-a
file_name = "{gene}-all-seqs"
full_name = f'/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/{file_name}'

# ^ (required character) '>'
# \S any non-whitespace character + (and all subsequent characters)
# [+-]? + or -, (optional)
# Ex. ">ENST00000350908.9_panTro4_1_16 56 0 0 chr7:115900699-115900866+"

pattern = re.compile(r"^>(\S+) (\d+) (\d+) (\d+) (\S+):(\d+)-(\d+)([+-]?)")

# search_species = ["hg38", "mm10", "panTro4", "ponAbe2", "gorGor3", "rheMac3"]          # primate species
search_species = ["pteAle1", "pteVam1", "myoDav1", "myoLuc2", "eptFus1", "oryLat2"]      # non-primate species

# To store in CSV
data = []
saved_species = set()  # To keep track of species already saved

with open(input_file, "r") as file:
    for line in file:
        line = line.strip()
        if line.startswith(">"):  # header line
            header = line
            try:
                sequence = next(file)  # get the sequence line
            except StopIteration:
                print("No sequence found for header:", header)
                continue
            match = pattern.match(header)
        else:
            continue    # Not a header line, ignore
        
        if enst_id[gene] in header:                                       # match Ensembl transcript ID
            # For searching for species
            # found_species = None
            found_species = True                                    # Looking for ALL 100 species
            # for sp in search_species:                             # iterate through the species we're looking for
            #      if sp in header:
            #           found_species = sp
            if found_species and match:
                # Store protein sequence information
                identifier = match.group(1)
                length_aa = match.group(2)
                start_phase = match.group(3)
                end_phase = match.group(4)
                chromosome = match.group(5)
                start_coord = match.group(6)
                end_coord = match.group(7)
                strand = match.group(8)
                aa_sequence = sequence

                # Get species identifier
                parts = identifier.split('_')
                species_name = parts[1]     # e.g. hg38 or panTro4
                exon_num = parts[2]        # e.g. _species_1_16

                # Add to data
                data.append((species_name, exon_num, length_aa, start_phase, end_phase, chromosome, start_coord, 
                            end_coord, strand, aa_sequence))
                
                saved_species.add(species_name)

print(f"Found {len(data)} sequences for {enst_id[gene]} in {len(saved_species)} species.")

df = pd.DataFrame(data, columns = [
     'Species', 'Number', 'AA Len', 'Start Phase', 'End Phase', 'Chromosome', 'Start Coord', 
     'End Coord', 'Strand', 'Seq'
])

df.to_csv(full_name+".csv", index=False)

print("Successfully saved amino acid sequences")
