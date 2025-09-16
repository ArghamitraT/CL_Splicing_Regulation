import pandas as pd
import numpy as np
import zarr
from Bio import AlignIO
import numpy as np
import gzip

from utils import (
    load_exon_coordinates_from_Multiz,
    load_exon_metadata_from_ASCOT,
    add_parsed_coordinates,
    find_overlaps
)


file_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/dummy_cassette_exons.csv'
maf_file_template = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/mafs/"
ref_species = "hg19.chr3"

df_ascot = load_exon_metadata_from_ASCOT(file_path)
exons_df = add_parsed_coordinates(df_ascot)

exons_df = exons_df.sort_values(by=["chromosome", "exon_start_parsed"])

# (AT)
# for chrom, group in exons_df.groupby('chromosome'):
chrom = "chr3"
group = exons_df[exons_df["chromosome"] == chrom]
print(f"🧬 Processing {len(group)} exons for chromosome: {chrom}")

# (AT)

maf_path = maf_file_template+f"{chrom}.maf.gz"
ref_prefix = f"{ref_species}.{chrom}"

with gzip.open(maf_path, "rt") as maf_file:  # "rt" is for reading as text

    msa_found = False
    alignments = AlignIO.parse(maf_file, "maf")
    
    # Flag to indicate if we found a relevant block
    msa_found = False

    # for alignment in alignments:
    for block_index, alignment in enumerate(alignments):
        # print(alignment)  # Process your alignment as needed
        # for block_index, alignment in enumerate(alignments):
        # Search for the target species in the alignment block
        for record in alignment:
            if ref_species in record.id:
                record_start = record.annotations.get('start', None)
                record_length = len(record.seq)
                record_strand = record.annotations.get('strand', None)
                
                # Calculate the end position based on strand
                if record_strand == 1:
                    record_end = record_start + record_length
                else:
                    record_end = record_start - record_length



                ######### need to check the logic of exon overlap #########
                
                # 🔁 Loop over all exons in this chromosome
                for _, exon in group.iterrows():
                    start_pos, end_pos, strand = (
                        exon["exon_start"],
                        exon["exon_end"],
                        exon["exon_strand"],
                    )
                    if (record_start >= end_pos and record_end <= start_pos) or \
                    (record_start <= end_pos and record_end >= start_pos) or \
                    (end_pos >= record_start >= start_pos and record_end <= start_pos) or \
                    (record_start >= end_pos and end_pos >= record_end >= start_pos):

                        msa_found = True
                        print(f"MSA Block for position {start_pos}-{end_pos} on strand {strand}:\n")
                        for rec in alignment:
                            species_id = rec.id
                            aligned_seq = rec.seq
                            
                            # Calculate the sequence segment's relative indices
                            start_index = start_pos - record_start if strand == "+" else record_start - end_pos
                            end_index = start_index + (end_pos - start_pos)
                            
                            # Extract the sequence segment
                            if record_strand == -1:
                                msa_segment = aligned_seq[start_index:end_index].reverse_complement()
                            else:
                                msa_segment = aligned_seq[start_index:end_index]
                            
                            print(f"{species_id}: {msa_segment}")
                        
                        print("\n")
                        break  # Found the relevant block, no need to search further
                    else:
                        print(f"not found in {block_index}: start {record_start} end {record_end}")

                    

                
           