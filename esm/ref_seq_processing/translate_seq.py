from Bio.Seq import Seq
import pickle
import argparse
import os

class Config:
    
    enst_id = {"foxp2": "ENST00000350908.9", "brca2": "ENST00000380152.7", "hla-a": "ENST00000376809.10", "tp53": "ENST00000269305.8"}
    def __init__(self, gene):
        self.gene = gene
        self.code = Config.enst_id[gene]
        self.input_file = '/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/from_ref_seqs/exon_nuc_seq.pkl'
        self.output_file = '/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/from_ref_seqs/{self.gene}_aa.json'

def main(cfg: Config):
    with open(cfg.input_file, "rb") as file:
        data = pickle.load(file)
    
    aa_dict = {}
    for species in data[cfg.code].keys():
        nuc_seq = Seq(data[cfg.code][species])
        aa_dict[species] = nuc_seq.translate()
        
        # TODO


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert nucleotide sequences to amino acid sequences")
    parser.add_argument(
        "--gene", 
        type=str,
        choices=['foxp2', 'brca2', 'hla-a', 'tp53'],
        required=True
    )
    args = parser.parse_args()
    cfg = Config(args.gene)
    main(cfg.gene)
