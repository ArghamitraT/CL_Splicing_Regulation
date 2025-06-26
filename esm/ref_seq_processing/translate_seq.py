from Bio.Seq import Seq
import pickle
import argparse
import pandas
import json

class Config:
    
    enst_id = {"foxp2": "ENST00000350908", "brca2": "ENST00000380152", "hla-a": "ENST00000376809", "tp53": "ENST00000269305"}
    exons_in_gene = {"foxp2": 16, "brca2": 27, "hla-a": 8, "tp53": 10}
    def __init__(self, gene):
        self.gene = gene
        self.code = Config.enst_id[gene]
        self.num_exons = Config.exons_in_gene[gene]
        self.input_file = '/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/from_ref_seqs/exon_nuc_seq.pkl'
        self.output_dir = f'/gpfs/commons/home/nkeung/cl_splicing/esm/processed_data/from_ref_seqs'

def get_exon_number(code: str):
    parts = code.split('_')
    return int(parts[1])


def main(cfg: Config):
    with open(cfg.input_file, "rb") as file:
        data = pickle.load(file)        # data[exon code][species] = string
    
    print("Type of data: ", type(data))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict, got {type(data)} instead")
    # Keep only this chosen gene, ignore all others
    filtered_exons = {key: value for (key, value) in data.items() if cfg.code in key}

    # Reshape dictionary to group by species
    grouped_exons = {}
    # for exon_code, species_dict in sorted_exons.items():
    for exon_code, species_dict in filtered_exons.items():
        for species, sequence in species_dict.items():
            if species not in grouped_exons:
                grouped_exons[species] = {}
            grouped_exons[species][get_exon_number(exon_code)] = sequence

    # Save exon border information
    csv_data = []

    # Saving full amino acid sequences
    aa_dict = {}
    for species in grouped_exons:
        nuc_seq = ""
        leftover = ""
        start_phase = 0
        end_phase = 0
        for i in range(1, cfg.num_exons + 1):
            if i not in grouped_exons[species]:
                print(f"Exon {i} not found in {species}")
                csv_data.append((species, i, start_phase, end_phase, ""))        # Amino acid length is 0
                continue

            sequence = grouped_exons[species][i]
            if sequence in [None, "", (None, None)]:
                print(f"No sequence found for {species} exon {i}")
                csv_data.append((species, i, start_phase, end_phase, ""))        # Aminio acid length is 0
                continue

            # Calculate exon phase information
            sequence = leftover + sequence
            num_codons = len(sequence) // 3
            in_frame = sequence[:num_codons * 3]
            leftover = sequence[num_codons * 3:]        # Any nucleotides not within frame
            end_phase = len(leftover)
            exon_aa = Seq(in_frame).translate()
            csv_data.append((species, i , start_phase, end_phase, str(exon_aa).replace("*", "")))

            nuc_seq += grouped_exons[species][i]

        if len(leftover) != 0:
            print(f"Warning: {len(leftover)} unused nucleotides for {species}")
        
        aa_seq = str(Seq(nuc_seq).translate()).replace("*", "")     # Trim stop codons
        aa_dict[species] = aa_seq

    aa_list = [(key, str(value)) for key, value in aa_dict.items()]
    with open(f"{cfg.output_dir}/{cfg.gene}-full-stitched.json", "w") as f:
        json.dump(aa_list, f)

    # Saving info in CSV for exon pooling
    df = pandas.DataFrame(csv_data, columns = ["Species", "Number", "Start Phase", "End Phase", "Seq"])
    df.to_csv(f"{cfg.output_dir}/{cfg.gene}-all-seqs.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grab specified gene and translate to amino acid")
    parser.add_argument(
        "--gene", 
        type=str,
        choices=['foxp2', 'brca2', 'hla-a', 'tp53'],
        default="tp53",
        # required=True
    )
    args = parser.parse_args()
    cfg = Config(args.gene)
    main(cfg)
