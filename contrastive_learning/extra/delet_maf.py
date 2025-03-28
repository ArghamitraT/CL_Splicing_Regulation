from Bio import AlignIO
import gzip

# Define the range to delete
start_pos = 43125000  # Replace with your start position
end_pos = 43126000    # Replace with your end position
target_chrom = "chr17"  # Replace with your target chromosome

maf_file_path = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/chr17.maf.gz"  # Input MAF file
output_file_path = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/chr17_short.maf"  # Output file after removing blocks

# Open the input and output MAF files
with gzip.open(maf_file_path, "rt") as maf_file:  # "rt" is for reading as text
    
    with open(output_file_path, "w") as output_file:
        alignments = AlignIO.parse(maf_file, "maf")

        for alignment in alignments:
            keep_block = False
            for record in alignment:
                chrom = record.id.split('.')[1]  # Extract chromosome
                start = record.annotations['start']
                length = record.annotations['size']
                strand = record.annotations['strand']
                
                # Calculate the end position
                end = start + length if strand == 1 else start - length
                MS = start
                ME = end
                SP = start_pos
                EP = end_pos
                # Check if this block overlaps with the deletion range
                if chrom == target_chrom:
                    if strand == 1:  # Positive strand
                        if (MS<=SP and ME>=EP) or \
                            (MS>=SP and ME<=EP) or \
                                (MS<=SP and SP<=ME<=EP) or \
                                (SP<=MS<=EP and ME>EP):
                            keep_block = True
                            break
                    else:  # Negative strand
                        if (MS<=SP and ME>=EP) or \
                            (MS>=SP and ME<=EP) or \
                                (EP<=MS<=SP and ME<=EP) or \
                                    (EP<=ME<=SP and MS>=SP):
                            keep_block = True
                            break
                # print(f"deleted {start} to {end}")
            
            # If the block doesn't fall in the deletion range, write it to the output file
            if keep_block:
                print(f"keeping {start} to {end}")

                AlignIO.write(alignment, output_file, "maf")
                