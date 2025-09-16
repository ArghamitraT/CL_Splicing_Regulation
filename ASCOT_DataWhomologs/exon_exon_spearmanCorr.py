
from utils import load_csv, get_tissue_PSI_ASCOT, compute_spearman_corr, save_matrix, compress_corr_matrix
import time

def main():
    start = time.time()
    # File paths
    file_name = "test"  # "train", "val", "test", "variable"
    file_ascot = f'/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/{file_name}_cassette_exons.csv'
    file_path = file_ascot
    output_path = f"/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/{file_name}_ExonExon_spearmanCorr.pkl"
    
    # Load exon file
    df = load_csv(file_path)

    # Get tissue expression matrix
    expr_matrix = get_tissue_PSI_ASCOT(df)
    exon_ids = df["exon_id"]

    # Compute Spearman correlation
    corr_df = compute_spearman_corr(expr_matrix, exon_ids)
    # corr_df = compress_corr_matrix(corr_df)
    
    # Save
    save_matrix(corr_df, output_path)
    print(f"✅ Correlation matrix saved to {output_path}")

    end = time.time()
    print(f"✅ Total runtime: {end - start:.2f} seconds")



if __name__ == "__main__":
    main()
