
from utils import load_exon_metadata_from_ASCOT, add_parsed_coordinates, save_matrix, process_maf_for_exons
import time





def main():
    start = time.time()
    # File paths
    file_name = "test"  # "train", "val", "test", "variable"
    file_ascot = f'/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/{file_name}_cassette_exons.csv'
    file_path = file_ascot
    output_path = f"/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/{file_name}_ExonExon_spearmanCorr.csv"
    
    # Load exon file
    df_ascot = load_exon_metadata_from_ASCOT(file_path)
    df_ascot = add_parsed_coordinates(df_ascot)

    maf_file_path = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/mafs/"
    ref_species = "hg38"
    process_maf_for_exons(df_ascot, maf_file_path, ref_species) 

    # Save
    # save_matrix(corr_df, output_path)
    # print(f"✅ Correlation matrix saved to {output_path}")

    end = time.time()
    print(f"✅ Total runtime: {end - start:.2f} seconds")



if __name__ == "__main__":
    main()
