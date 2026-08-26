import os
import argparse
import pandas as pd
import pickle

data_dir = "/mnt/home/nlk2136/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/chromSplit"

def format_fasta(seq_id, species, sequence):
    """Format as FASTA entry"""
    return f">{seq_id}__{species}\n{sequence}\n"


def write_fasta_buffered(data, output_file, buffer_size=500):
    """
    Iterate through data dict and write FASTA with buffering
    """
    buffer = []
    seq_count = 0
    exon_count = 0

    with open(output_file, 'w', buffering=1024*1024) as fasta_out:

        # ENST00000696069.1_6_7: {'hg38': {'5p': 'GTAAGTTTTG', 'exon': 'GGG', '3p': 'agatggc'} }

        for exon_id, record in data.items():
            for species, sequences in record.items():
                sequence = sequences['5p'] + sequences['exon'] + sequences['3p']
                
                buffer.append(format_fasta(exon_id, species, sequence))
                seq_count += 1
            
            exon_count += 1
            # Write in batches
            if len(buffer) >= buffer_size:
                fasta_out.writelines(buffer)
                buffer.clear()
                print(f"\tWrote {seq_count} sequences...")
        
        # Final flush for remaining entries
        if buffer:
            fasta_out.writelines(buffer)
        
        print(f"Completed! Total sequences written: {seq_count}")
        print(f"Total exons: {exon_count}\n")


def main(target_split):
    file_name = os.path.join(data_dir, f"{target_split}_chr_merged_filtered_min30Views.pkl")

    with open(file_name, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded {target_split}. {len(list(data.keys()))} exons to process")

    output_file = os.path.join(data_dir, f"{target_split}_hg38_sequences.fa")

    # Stream through data with buffered writes
    write_fasta_buffered(data, output_file, buffer_size=5000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test", "sample"], required=True)
    args = parser.parse_args()

    target_split = args.split

    main(target_split)