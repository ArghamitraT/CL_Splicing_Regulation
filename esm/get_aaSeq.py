"""
Searches for a given gene and its amino acid alignment with other species.
"""

# import pandas as pd
# import gzip
import re
import pickle

# alignment_file = "knownGene.exonAA.fa"
alignment_file = "test_truncated.fa"
input_file = "/gpfs/commons/home/nkeung/data/" + alignment_file

# ^ (required character) '>'
# \S any non-whitespace character + (and all subsequent characters)
# [+-]? + or -, (optional)
# Ex. ">ENST00000350908.9_panTro4_1_16 56 0 0 chr7:115900699-115900866+"

pattern = re.compile(r"^>(\S+) (\d+) (\d+) (\d+) (\S+):(\d+)-(\d)[+-]?")

enst_id = "ENST00000350908.9"
species = ["hg38"]
seq_length = "56"       # corresponds to match.group(2)

found_Match = False
with open(input_file, "r") as file:
    lines = file.readlines()

for header, sequence in zip(lines[::2], lines[1::2]):
        match = pattern.match(header)
        if enst_id in header:
            if any(sp in header for sp in species):
                # if match.group(2) == seq_length:
                print(sequence)
        
