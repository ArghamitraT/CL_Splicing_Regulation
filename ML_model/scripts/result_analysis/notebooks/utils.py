def load_ground_truth(filepath: str) -> (pd.DataFrame, list):
    """
    Loads the ground truth PSI CSV file and returns the
    wide DataFrame and the list of tissue columns.
    """
    print(f"Loading and processing Ground Truth from: {filepath}")
    df = pd.read_csv(filepath)
    
    # This column indexing is from your script
    meta_cols = [
        'exon_id', 'cassette_exon', 'alternative_splice_site_group', 'linked_exons',
        'mutually_exclusive_exons', 'exon_strand', 'exon_length', 'gene_type',
        'gene_id', 'gene_symbol', 'exon_location', 'exon_boundary',
        'chromosome', 'mean_psi', 'logit_mean_psi', 'chromosome.1'
    ]
    tissue_cols = [col for col in df.columns if col not in meta_cols]
    
    # --- NEW: Ensure PSI values are 0-1 ---
    # Check if max PSI is > 1.5, indicating 0-100 scale
    if not df[tissue_cols].empty and (df[tissue_cols].max(skipna=True).max(skipna=True) > 1.5):
        print("Ground Truth: Detected PSI values > 1.5, converting from 0-100 scale.")
        df[tissue_cols] = df[tissue_cols] / 100.0
    
    return df, list(tissue_cols)