"""
Given .bed file as input and .bw file as phastacon score, it calculates the score
"""

import os
import subprocess

def run_bigWigAverageOverBed(input_bw, input_dir, output_dir):
    # # Ensure the output directory exists
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # Get a list of all files in the input directory
    # Construct the output file name based on the input file name
    file = os.path.splitext(os.path.basename(input_dir))[0]
    output_tsv = os.path.join(output_dir, f"{file}.tsv")
    input_bed = input_dir

    # Construct and run the command
    command = f"bigWigAverageOverBed {input_bw} {input_bed} {output_tsv}"
    subprocess.run(command, shell=True, check=True)
    print(f"Processed {file}")
    
    # for file in files:
    #     if file.endswith('.bed'):  # Only process BED files
    #         # Construct the output file name based on the input file name
    #         output_tsv = os.path.join(output_dir, f"{file}.tsv")
    #         input_bed = os.path.join(input_dir, file)

    #         # Construct and run the command
    #         command = f"bigWigAverageOverBed {input_bw} {input_bed} {output_tsv}"
    #         subprocess.run(command, shell=True, check=True)
    #         print(f"Processed {file}")

# Example usage
input_bw = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/existing_score/hg38.phastCons100way.bw"  # Path to the .bw file
# input_dir = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/existing_score/knownGene.multiz100way.exonNuc_exon_intron_positions_intron.bed"  # Directory containing BED files
# input_dir = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/dummy_exon_intron_positions_shrt_exon.bed"  # Directory containing BED files
input_dir = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/new_run/19_chr/bed_files/knownGene.multiz100way.exonNuc_exon_intron_positions_chr19_exon.bed'  # Replace with actual path

output_dir = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/phastcon_score/"  # Directory to save TSV output files

run_bigWigAverageOverBed(input_bw, input_dir, output_dir)