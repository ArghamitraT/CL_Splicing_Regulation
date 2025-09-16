import zarr
from Bio import AlignIO
import numpy as np
import gzip

# Input .maf file and output .zarr file paths
# maf_file_path = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/chr1_GL383518v1_alt.maf"
maf_file_path = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/mafs/chr17.maf.gz"
# zarr_file = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/chr1_GL383518v1_alt.zarr"
target_species = "hg19.chr17"  # The species you are searching for
start_pos = 43125271	  # Replace with the start position you're interested in
end_pos = 43125364  # Replace with the end position you're interested in
strand = "-"  # Specify the strand ('+' or '-')

# Open the compressed .maf.gz file
# with gzip.open(maf_file_path, "rt") as maf_file:  # "rt" is for reading as text
#     alignments = AlignIO.parse(maf_file, "maf")
    
#     for alignment in alignments:
#         print(alignment)  # Process your alignment as needed

with gzip.open(maf_file_path, "rt") as maf_file:  # "rt" is for reading as text
    alignments = AlignIO.parse(maf_file, "maf")
    
    # Flag to indicate if we found a relevant block
    msa_found = False

    # for alignment in alignments:
    for block_index, alignment in enumerate(alignments):
        # print(alignment)  # Process your alignment as needed
        # for block_index, alignment in enumerate(alignments):
        # Search for the target species in the alignment block
        for record in alignment:
            if target_species in record.id:
                record_start = record.annotations.get('start', None)
                record_length = len(record.seq)
                record_strand = record.annotations.get('strand', None)
                
                # Calculate the end position based on strand
                if record_strand == 1:
                    record_end = record_start + record_length
                else:
                    record_end = record_start - record_length

                # Check if the alignment overlaps the specified region and strand
                # if (strand == "+" and record_strand == 1 and start_pos >= record_start and end_pos <= record_end) or \
                #     (strand == "-" and record_strand == -1 and start_pos <= record_start and end_pos >= record_end):
                # if (start_pos >= record_start and end_pos <= record_end) or \
                #     (start_pos <= record_start and end_pos >= record_end) or\
                #      (start_pos <= record_start and end_pos >= record_end) or\
                #          (start_pos >= record_start and end_pos <= record_end) :
                
                # if (record_start<=start_pos and record_end>=record_end) or \
                #     (record_start>=start_pos and record_end<=record_end) or\
                #     (start_pos<=record_start<=end_pos and record_end>=record_end) or\
                #         (record_start<=start_pos and start_pos<=record_end<=record_end) :
                    
                if (record_start >= end_pos and record_end <= start_pos) or \
                    (record_start <= end_pos and record_end >= start_pos) or \
                    (end_pos >= record_start >= start_pos and record_end <= start_pos) or \
                    (record_start >= end_pos and end_pos >= record_end >= start_pos):

                    msa_found = True
                    print(f"MSA Block for position {start_pos}-{end_pos} on strand {strand}:\n")
                    
                    # Loop through all species in the alignment block to extract the MSA
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

    if not msa_found:
        print(f"No MSA block found for position {start_pos}-{end_pos} on strand {strand}.")


