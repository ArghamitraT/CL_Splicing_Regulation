from utils import (
    load_exon_metadata_from_ASCOT,
    load_spearmanCorrFile,
    high_corr_exons,
    filter_exons_ASCOT,
    save_csv,
    expected_high_corr_per_exon
)

import pandas as pd

def main(file_spCorr: str, file_ASCOT: str, output_file: str = "overlap_results.csv"):
    

    corr_df = load_spearmanCorrFile(file_spCorr)
    # corr_df = load_exon_metadata_from_ASCOT(file_spCorr)
    # high_corr_exonset = high_corr_exons(corr_df, sp_threshold=0.9, exon_similarity_threshold=500)
    # ascot_df = load_exon_metadata_from_ASCOT(file_ASCOT)
    expected_high_corr_per_exon(corr_df, threshold=0.8)
    # filtered_df = filter_exons_ASCOT(ascot_df, high_corr_exonset)
    # save_csv(filtered_df, output_file)


    

if __name__ == "__main__":
    
    division = 'val'  # 'train', 'val', 'test'
    file_ascot = f'/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/{division}_cassette_exons.csv'
    file_spCorr = f'/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/{division}_ExonExon_spearmanCorr.pkl'
    output_file = f'/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/{division}_cassette_exons_filtered.csv'
    # file_spCorr = f'/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/dummy_ExonExon_spearmanCorr.csv'
    
    # file = pd.read_csv('/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/gtex_psi.csv')
    # print(len(file))
    main(file_spCorr, file_ascot, output_file)