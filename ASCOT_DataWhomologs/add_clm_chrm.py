from utils import (
    load_csv,
    remove_allOther_species_multiz,
    load_exon_metadata_from_ASCOT,
    add_parsed_coordinates,
    find_overlaps,
    save_csv
)

import pandas as pd

def main(file1: str, file2: str, output_file: str = "overlap_results.csv"):
    
    # # df_ncbi = load_exon_coordinates_from_Multiz("/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/ncbiRefSeq.multiz100way.exonNuc_exon_intron_positions.csv")
    # # df_known = load_exon_coordinates_from_Multiz("/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/knownGene.multiz100way.exonNuc_exon_intron_positions.csv")
    # df_ncbi = load_csv("/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/ncbiRefSeq.multiz100way.exonNuc_exon_intron_positions.csv")
    # df_known = load_csv("/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/knownGene.multiz100way.exonNuc_exon_intron_positions.csv")
    # df_multiz_old = pd.concat([df_known, df_ncbi], ignore_index=True)

    # df_multiz = remove_allOther_species_multiz(df_multiz_old)
    
    # df_multiz = load_exon_coordinates_from_Multiz(file1)
    df_ascot = load_exon_metadata_from_ASCOT(file2)
    df_ascot = add_parsed_coordinates(df_ascot)
    modified_ascot = df_ascot.iloc[:, :-2]
    modified_ascot.rename(columns={"chromosome_parsed": "chromosome"}, inplace=True)


    # modified_ascot = df_ascot[df_ascot["chromosome"] == "chr3"]

    # matches = find_overlaps(df_multiz, df_ascot, whichPrime=5)
    # exon_ids_to_keep = matches[df_multiz.columns.tolist() + ["exon_id"]].rename(columns={"exon_id": "ascot_exon_id"})["Exon Name"].unique()
    # filtered_df_multiz = df_multiz_old[df_multiz_old["Exon Name"].isin(exon_ids_to_keep)].reset_index(drop=True)

    # # Keep mapping Exon Name -> ascot_exon_id from matches
    # mapping = matches[["Exon Name", "exon_id"]].rename(columns={"exon_id": "ascot_exon_id"})
    # # Merge mapping into filtered_df_multiz
    # filtered_df_multiz = filtered_df_multiz.merge(mapping, on="Exon Name", how="left")

    # # Keep all df_multiz columns + exon_id (renamed ascot_exon_id)
    save_csv(modified_ascot, output_file)
    # overlaps.to_csv(output_file, index=False)
    # print(f"✅ Found {len(overlaps)} overlaps. Results saved to {output_file}")


if __name__ == "__main__":
    # file_multiz = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/knownGene.multiz100way.exonNuc_exon_intron_positions.csv'
    
    file_name = "variable"
    file_multiz = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/ncbiRefSeq.multiz100way.exonNuc_exon_intron_positions.csv'
    file_ascot = f'/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/{file_name}_cassette_exons.csv'
    output_file = f'/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/{file_name}_cassette_exons_modified.csv'

    # file = pd.read_csv('/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/gtex_psi.csv')
    # print(len(file))
    main(file_multiz, file_ascot, output_file)
