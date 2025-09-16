import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# def corr_vs_nan(expr_matrix: pd.DataFrame, corr_df: pd.DataFrame):
#     """
#     Make a scatter plot of Spearman correlation vs. max NaN tissue count.
    
#     Args:
#         expr_matrix: DataFrame (N_exons × N_tissues), with NaNs for missing PSI
#         corr_df: Spearman correlation matrix (N×N)
#     """
    
#     nan_counts = expr_matrix.isna().sum(axis=1)
#     nan_counts.index = expr_matrix.index  # exon IDs

#     # Make sure index and columns are not both called "exon_id"
#     corr_df = corr_df.copy()
#     corr_df.index.name = "exon1"
#     corr_df.columns.name = "exon2"

#     # Convert correlation matrix to long form (exclude diagonal)
#     mask = ~np.eye(len(corr_df), dtype=bool)
#     corr_long = (
#         corr_df.where(mask)
#                .stack()
#                .reset_index(name="spearman_corr")
#     )

#     # Max NaN count for each pair
#     corr_long["max_nan"] = corr_long.apply(
#         lambda row: max(nan_counts.loc[row["exon1"]], nan_counts.loc[row["exon2"]]),
#         axis=1
#     )

#     # Plot
#     plt.figure(figsize=(8, 6))
#     plt.scatter(corr_long["max_nan"], corr_long["spearman_corr"], alpha=0.2, s=5)
#     plt.xlabel("Max NaN tissue count between exon pair")
#     plt.ylabel("Spearman correlation")
#     plt.title("Correlation vs. Max NaN Tissue Count")
#     plt.show()

#     return corr_long


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def corr_vs_nan(expr_matrix: pd.DataFrame, corr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Make a scatter plot of Spearman correlation vs. max NaN tissue count.

    Args:
        expr_matrix: DataFrame (N_exons × N_tissues), with NaNs for missing PSI
        corr_df: Spearman correlation matrix (N×N)

    Returns:
        corr_long: DataFrame with columns [exon1, exon2, spearman_corr, max_nan]
    """
    # Count NaNs per exon (index = exon_id)
    nan_counts = expr_matrix.isna().sum(axis=1)

    # Ensure correlation matrix index/cols are named properly
    corr_df = corr_df.copy()
    corr_df.index.name = "exon1"
    corr_df.columns.name = "exon2"

    # Long form correlation matrix (exclude diagonal)
    mask = ~np.eye(len(corr_df), dtype=bool)
    corr_long = (
        corr_df.where(mask)
               .stack()
               .reset_index(name="spearman_corr")
    )

    # Vectorized lookup for nan counts
    corr_long["nan_exon1"] = corr_long["exon1"].map(nan_counts)
    corr_long["nan_exon2"] = corr_long["exon2"].map(nan_counts)
    corr_long["max_nan"] = corr_long[["nan_exon1", "nan_exon2"]].max(axis=1)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(
        corr_long["max_nan"],
        corr_long["spearman_corr"],
        alpha=0.2,
        s=5,
    )
    plt.xlabel("Max NaN tissue count between exon pair")
    plt.ylabel("Spearman correlation")
    plt.title("Correlation vs. Max NaN Tissue Count")
    plt.savefig("/gpfs/commons/home/atalukder/Contrastive_Learning/code/ASCOT_DataWhomologs/figures/corr_vs_nan.png", dpi=300)
    plt.show()

    return corr_long



from utils import (
    load_exon_metadata_from_ASCOT,
    load_spearmanCorrFile,
)


def get_tissue_PSI_ASCOT(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only the tissue expression values from exon metadata DataFrame.
    - Tissue columns are assumed to be between 'exon_boundary' and 'chromosome'.
    - Replace -1.0 with NaN so they are ignored in correlation.
    - Use exon_id as the index for consistent alignment.
    """
    # Find column positions dynamically
    start_idx = df.columns.get_loc("exon_boundary") + 1
    end_idx = df.columns.get_loc("chromosome")
    tissue_cols = df.columns[start_idx:end_idx]

    # Extract PSI values
    expr_matrix = df[["exon_id"] + list(tissue_cols)].copy()

    # Convert to float and set index = exon_id
    expr_matrix[tissue_cols] = expr_matrix[tissue_cols].astype(float)
    expr_matrix = expr_matrix.set_index("exon_id")

    # Replace -1.0 with NaN
    expr_matrix = expr_matrix.replace(-1.0, np.nan)

    return expr_matrix


import pandas as pd

def main(file_spCorr: str, file_ASCOT: str, output_file: str = "overlap_results.csv"):
    

    corr_df = load_spearmanCorrFile(file_spCorr)
    
    ascot_df = load_exon_metadata_from_ASCOT(file_ASCOT)

    
    ascot_df = get_tissue_PSI_ASCOT(ascot_df)

    results = corr_vs_nan(ascot_df, corr_df)


    

if __name__ == "__main__":
    
    division = 'val'  # 'train', 'val', 'test'
    file_ascot = f'/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/{division}_cassette_exons.csv'
    file_spCorr = f'/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/{division}_ExonExon_spearmanCorr.pkl'
    output_file = f'/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/{division}_cassette_exons_filtered.csv'
    # file_spCorr = f'/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/dummy_ExonExon_spearmanCorr.csv'
    
    # file = pd.read_csv('/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/gtex_psi.csv')
    # print(len(file))
    main(file_spCorr, file_ascot, output_file)
