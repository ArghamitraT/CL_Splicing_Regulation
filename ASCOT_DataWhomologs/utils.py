import pandas as pd
import numpy as np


def save_csv(df: pd.DataFrame, output_path: str):
    """
    Save DataFrame to CSV.
    """
    df.to_csv(output_path, index=False)


def remove_allOther_species_multiz(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load exon coordinate file (species-based).
    """
    # return pd.read_csv(file_path)
    df = df[df["Species Name"] == "hg38"].reset_index(drop=True)
    print(f"Loaded {len(df)} exons for hg38 from Multiz data.")
    return df

def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load exon metadata file (with PSI/tissue expression).
    """
    return pd.read_csv(file_path)



def load_exon_metadata_from_ASCOT(file_path: str) -> pd.DataFrame:
    """
    Load exon metadata file (with PSI/tissue expression).
    """
    return pd.read_csv(file_path)


def parse_location(location: str):
    """
    Parse genomic location like 'chr19:58352098-58352184' → (chrom, start, end).
    """
    chrom, coords = location.split(":")
    start, end = map(int, coords.split("-"))
    return chrom, start, end


def add_parsed_coordinates(df: pd.DataFrame, location_col: str = "exon_location") -> pd.DataFrame:
    """
    Add parsed chromosome, start, end columns to metadata DataFrame.
    """
    parsed = df[location_col].apply(parse_location)
    df["chromosome_parsed"] = parsed.apply(lambda x: x[0])
    df["exon_start_parsed"] = parsed.apply(lambda x: x[1])
    df["exon_end_parsed"] = parsed.apply(lambda x: x[2])
    return df


def find_overlaps(df_multiz, df_ascot, whichPrime = 3) -> pd.DataFrame:
    """
    Find overlapping exons between two datasets.
    df1: coordinates (first file)
    df2: metadata with parsed locations (second file)
    """
    # if whichPrime == 3:
    plus_matches_3p = pd.merge(
        df_ascot[df_ascot['exon_strand'] == '+'],
        df_multiz,
        left_on=['chromosome_parsed', 'exon_start_parsed'],
        right_on=['Chromosome', 'Exon Start']
    ).drop_duplicates(subset=['Chromosome', 'Exon Start'])


    minus_matches_3p = pd.merge(
        df_ascot[df_ascot['exon_strand'] == '-'],
        df_multiz,
        left_on=['chromosome_parsed', 'exon_end_parsed'],
        right_on=['Chromosome', 'Exon End']
    ).drop_duplicates(subset=['Chromosome', 'Exon End'])
    
    # elif whichPrime == 5:
    plus_matches_5p = pd.merge(
        df_ascot[df_ascot['exon_strand'] == '+'],
        df_multiz,
        left_on=['chromosome_parsed', 'exon_end_parsed'],
        right_on=['Chromosome', 'Exon End']
    ).drop_duplicates(subset=['Chromosome', 'Exon End'])

    minus_matches_5p = pd.merge(
        df_ascot[df_ascot['exon_strand'] == '-'],
        df_multiz,
        left_on=['chromosome_parsed', 'exon_start_parsed'],
        right_on=['Chromosome', 'Exon Start']
    ).drop_duplicates(subset=['Chromosome', 'Exon Start'])

    # Concatenate all matches
    matches = pd.concat([plus_matches_3p, minus_matches_3p, plus_matches_5p, minus_matches_5p], ignore_index=True)

    # Drop duplicates based on Chromosome + coordinates
    matches = matches.drop_duplicates(subset=['Chromosome', 'Exon Start', 'Exon End'])
    total_unique_boundaries = len(df_ascot)

    match_percentage = (len(matches) / total_unique_boundaries) * 100
    print(f"Match Percentage: {match_percentage}")

    return matches


    
def get_tissue_PSI_ASCOT(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only the tissue expression values from exon metadata DataFrame.
    Assumes tissue columns start after 'exon_boundary' and before the last 'chromosome' column.
    """
    # Find column positions dynamically
    start_idx = df.columns.get_loc("exon_boundary") + 1
    end_idx = df.columns.get_loc("chromosome")
    tissue_cols = df.columns[start_idx:end_idx]
    
    # Extract and convert to float
    expr_matrix = df[tissue_cols].astype(float)
    
    # Replace -1.0 values with NaN
    expr_matrix = expr_matrix.replace(-1.0, np.nan)
    
    return expr_matrix


def compute_spearman_corr(expr_matrix: pd.DataFrame, exon_ids: pd.Series) -> pd.DataFrame:
    """
    Compute Spearman correlation between exons based on tissue profiles.
    
    Args:
        expr_matrix: DataFrame (N_exons × N_tissues)
    
    Returns:
        corr_df: Spearman correlation matrix (N_exons × N_exons)
    """
    corr_df = expr_matrix.T.corr(method="spearman")
    
    # Rename rows and columns to exon_id
    corr_df.index = exon_ids
    corr_df.columns = exon_ids
    
    return corr_df


def compress_corr_matrix(corr_df: pd.DataFrame):
    """
    Convert correlation matrix to long-form (pairs + corr).
    Only keeps upper triangle (i<j).
    """
    exon_ids = corr_df.index
    tril_mask = np.tril(np.ones(corr_df.shape), k=0).astype(bool)
    
    corr_df = corr_df.mask(tril_mask)  # mask lower triangle & diagonal
    
    # Convert to long format
    corr_long = corr_df.stack().reset_index()
    corr_long.columns = ["exon1", "exon2", "spearman_corr"]
    
    return corr_long


def save_matrix(corr_df: pd.DataFrame, output_path: str):
    """
    Save correlation matrix to CSV.
    """
    # corr_df.to_csv(output_path, index=True)
    corr_df.to_pickle(output_path)


def load_spearmanCorrFile(file_path: str) -> pd.DataFrame:
    """
    Load exon metadata file (with PSI/tissue expression).
    """
    return pd.read_pickle(file_path)


def expected_high_corr_per_exon(corr_df: pd.DataFrame, threshold: float = 0.8) -> float:
    """
    Compute the expected (average) number of exons per exon 
    that have correlation above threshold.
    
    Args:
        corr_df: Correlation matrix (exon × exon)
        threshold: cutoff value for "high correlation"
    
    Returns:
        avg_corr_partners: average number of correlated exons per exon
    """
    # Mask diagonal so self-correlations don't count
    corr = corr_df.copy()
    np.fill_diagonal(corr.values, np.nan)
    
    # Count how many exons each exon is correlated with above threshold
    counts = (corr >= threshold).sum(axis=1)
    
    # Average across all exons
    avg_corr_partners = counts.mean()
    
    print(f"Average number of exons per exon with corr ≥ {threshold}: {avg_corr_partners:.2f}")
    return avg_corr_partners



def high_corr_exons(corr_df: pd.DataFrame, sp_threshold: float = 0.8, exon_similarity_threshold: int = 500) -> set:
    
    """
    Load correlation data and return set of exons with corr >= threshold.
    Assumes correlation file is long-form: [exon1, exon2, spearman_corr].
    sp_threshold: Correlation threshold to consider "highly correlated".
    exon_similarity_threshold: Number of highly correlated exons to consider an exon "hyper-correlated". As there are a lot of -1 (NaN) values in the correlation matrix, we first remove exons that have more than this number of 1s (self-correlations) before applying the sp_threshold filter.
    values, exons with a lot of NaN values can end up being correlated with almost all exons just by chance. thus filter out those exons first. the number can be tuned. Eg, out of 1000 exons, if an exon has more than 500 exons with correlation 1, we consider it hyper-correlated and remove it first.
    """

    print(f"Original correlation matrix shape: {corr_df.shape}")

    # --- Step 1: Pre-filter to remove hyper-correlated exons ---
    # Count the number of '1s' in each row
    ones_per_exon = (corr_df == 1).sum(axis=1)

    # Identify exons to remove where the count of 1s exceeds the removal threshold
    exons_to_remove = ones_per_exon[ones_per_exon > exon_similarity_threshold].index
    num_removed = len(exons_to_remove)

    # Drop the identified rows and their corresponding columns
    corr_df = corr_df.drop(index=exons_to_remove, columns=exons_to_remove)

    print(f"Removed {num_removed} hyper-correlated exons (had > {exon_similarity_threshold} ones).")
    print(f"Filtered matrix shape: {corr_df.shape}")

    # --- Average correlation (off-diagonal only) ---
    off_diag = corr_df.where(~np.eye(len(corr_df), dtype=bool))
    avg_corr = off_diag.stack().mean()
    print(f"Average correlation (off-diagonal): {avg_corr:.4f}")

    # Create a boolean matrix where True indicates a correlation > threshold
    low_corr_mask = corr_df.values > sp_threshold

    # Set the diagonal to False to ignore self-correlations
    np.fill_diagonal(low_corr_mask, False)
    
    # Find the row and column indices where the condition is True
    involved_rows, involved_cols = np.where(low_corr_mask)
    
    # Combine all indices and find the unique ones
    all_involved_indices = np.concatenate([involved_rows, involved_cols])
    unique_involved_indices = np.unique(all_involved_indices)
    
    # Get the exon names from the DataFrame's index
    exon_names = corr_df.index.to_numpy()
    
    # Select the names of the involved exons using their indices
    involved_exon_names = exon_names[unique_involved_indices]

    print(f"Found {len(set(involved_exon_names))} high-corr exons to remove")

    # Return the names as a set for automatic de-duplication
    return set(involved_exon_names)


    
    # high_corr = corr_df[corr_df["spearman_corr"] >= threshold]
    
    # # Collect all exons that appear in high correlation pairs
    # exon_set = set(high_corr["exon1"]).union(set(high_corr["exon2"]))
    # print(f"Found {len(exon_set)} high-corr exons to remove")
    # return exon_set


def filter_exons_ASCOT(ascot_df: pd.DataFrame, high_corr_exonset: set) -> pd.DataFrame:

    """
    Remove exons that are highly correlated (>= threshold) from metadata CSV.
    """
    
    # Filter metadata
    filtered_df = ascot_df[~ascot_df["exon_id"].isin(high_corr_exonset)].reset_index(drop=True)
    return filtered_df
    
    


"""
# MAF processing functions


"""
# import gzip
# from Bio import AlignIO

# # def find_maf_block_for_exon(maf_file, ref_species, chrom, start_pos, end_pos, strand):
# #     """
# #     Search MAF for block overlapping given exon coordinates in ref_species.
# #     Returns the alignment block or None if not found.
# #     """
# #     with gzip.open(maf_file, "rt") as maf_file_handle:
# #         alignments = AlignIO.parse(maf_file_handle, "maf")
        
# #         for block in alignments:
# #             for record in block:
# #                 if ref_species in record.id and chrom in record.id:
# #                     record_start = record.annotations["start"]
# #                     record_len = len(record.seq)
# #                     record_strand = record.annotations["strand"]
# #                     record_end = record_start + record_len if record_strand == 1 else record_start - record_len

# #                     # overlap check
# #                     if not (end_pos < record_start or start_pos > record_end):
# #                         return block
# #     return None

# # def extract_species_coordinates(block, ref_species, start_pos, end_pos, strand, species_list=None):
# #     """
# #     Given an alignment block and exon coordinates, return start/end for species.

# #     If species_list is None, returns ALL species in the block.
# #     Otherwise, returns only the requested species (fills NA if missing).
# #     """
# #     results = {}
# #     available_species = {rec.id.split(".")[0]: rec for rec in block}

# #     if species_list is None:
# #         # Extract all species in block
# #         for sp, rec in available_species.items():
# #             aln_start = rec.annotations["start"]
# #             aln_len = len(rec.seq)
# #             aln_strand = rec.annotations["strand"]
# #             aln_end = aln_start + aln_len if aln_strand == 1 else aln_start - aln_len
# #             results[f"{sp}_start"] = aln_start
# #             results[f"{sp}_end"] = aln_end
# #     else:
# #         # Extract only requested species
# #         for sp in species_list:
# #             if sp in available_species:
# #                 rec = available_species[sp]
# #                 aln_start = rec.annotations["start"]
# #                 aln_len = len(rec.seq)
# #                 aln_strand = rec.annotations["strand"]
# #                 aln_end = aln_start + aln_len if aln_strand == 1 else aln_start - aln_len
# #                 results[f"{sp}_start"] = aln_start
# #                 results[f"{sp}_end"] = aln_end
# #             else:
# #                 results[f"{sp}_start"] = "NA"
# #                 results[f"{sp}_end"] = "NA"
# #     return results



# def _map_genomic_to_alignment_coords(ref_record, exon_start: int, exon_end: int) -> tuple[int | None, int | None]:
#     """
#     Maps genomic coordinates of an exon to alignment coordinates within a MAF block.

#     Args:
#         ref_record: The Bio.SeqRecord object for the reference species from the block.
#         exon_start: The genomic start coordinate of the exon.
#         exon_end: The genomic end coordinate of the exon.

#     Returns:
#         A tuple of (alignment_start, alignment_end) indices for slicing.
#     """
#     block_start = ref_record.annotations["start"]
#     ref_seq = str(ref_record.seq)
    
#     align_start_idx, align_end_idx = None, None
#     genomic_pos = block_start
    
#     for i, char in enumerate(ref_seq):
#         if char == '-':
#             continue # Gaps don't advance genomic position
        
#         # Check for the start of the exon
#         if align_start_idx is None and genomic_pos >= exon_start:
#             align_start_idx = i
        
#         # Check for the end of the exon
#         if genomic_pos >= exon_end - 1:
#             align_end_idx = i + 1
#             break
            
#         genomic_pos += 1
        
#     return align_start_idx, align_end_idx


# def extract_species_alignments(
#     block: AlignIO.MultipleSeqAlignment,
#     align_start: int,
#     align_end: int,
#     ref_strand: int,
#     exon_strand: str
# ) -> dict[str, str]:
#     """
#     Extracts a slice from all sequences in a block and handles strand correction.

#     Args:
#         block: The MultipleSeqAlignment block.
#         align_start: The starting index for slicing the alignment.
#         align_end: The ending index for slicing the alignment.
#         ref_strand: The strand of the reference sequence in the MAF block (1 or -1).
#         exon_strand: The strand of the exon ('+' or '-').

#     Returns:
#         A dictionary mapping each species name to its aligned sequence for the exon.
#     """
#     alignments = {}
    
#     # Determine if a reverse complement is needed
#     # This happens if the exon is on the '-' strand and the MAF block is on the '+' strand, or vice-versa.
#     needs_reverse_complement = (exon_strand == '-') != (ref_strand == -1)

#     for record in block:
#         species = record.id.split('.')[0]
#         # Slice the alignment to get the sequence corresponding to the exon
#         sliced_seq = record.seq[align_start:align_end]
        
#         if needs_reverse_complement:
#             # If strands are discordant, we must reverse complement the sequence
#             alignments[species] = str(Seq(str(sliced_seq)).reverse_complement())
#         else:
#             alignments[species] = str(sliced_seq)
            
#     return alignments

# # Assume the helper functions _map_genomic_to_alignment_coords and 
# # extract_species_alignments from the previous response exist here.

# def _process_single_maf(maf_path: str, exons_to_find: list[dict], ref_prefix: str):
#     """
#     Worker function to find exon overlaps within a single MAF file.

#     This is a generator that yields results as it finds them.

#     Args:
#         maf_path (str): The full path to a gzipped MAF file.
#         exons_to_find (list[dict]): A list of exon dictionaries to find in this file.
#         ref_prefix (str): The reference species prefix (e.g., "hg38.chr1").

#     Yields:
#         tuple[Any, dict]: A tuple containing the original exon index and the
#                           dictionary of species alignments.
#     """
#     try:
#         with gzip.open(maf_path, "rt") as maf_handle:
#             for block in AlignIO.parse(maf_handle, "maf"):
#                 if not exons_to_find:
#                     break  # Optimization: stop if all exons are found

#                 ref_record = next((r for r in block if r.id.startswith(ref_prefix)), None)
#                 if not ref_record:
#                     continue

#                 block_start = ref_record.annotations["start"]
#                 block_end = block_start + ref_record.annotations["size"]
#                 ref_strand = ref_record.annotations["strand"]

#                 remaining_exons = []
#                 for exon in exons_to_find:
#                     # Check for overlap between exon and MAF block
#                     if exon['start'] < block_end and block_start < exon['end']:
#                         a_start, a_end = _map_genomic_to_alignment_coords(
#                             ref_record, exon['start'], exon['end']
#                         )
#                         if a_start is not None and a_end is not None:
#                             species_aligns = extract_species_alignments(
#                                 block, a_start, a_end, ref_strand, exon['strand']
#                             )
#                             # Yield the original index and the extracted alignment data
#                             yield exon['idx'], species_aligns
#                     else:
#                         remaining_exons.append(exon)
                
#                 exons_to_find = remaining_exons

#     except FileNotFoundError:
#         print(f"⚠️ Warning: MAF file not found at '{maf_path}'.")
#     except Exception as e:
#         print(f"An error occurred while processing '{maf_path}': {e}")


# def process_maf_for_exons(
#     exons_df: pd.DataFrame,
#     maf_file_template: str,
#     ref_species: str
# ) -> pd.Series:
#     """
#     Main controller function to find and extract alignments for exons in a DataFrame.

#     This function processes the dataframe chromosome by chromosome, opening each
#     MAF file only once to find all corresponding exon alignments.

#     Args:
#         exons_df (pd.DataFrame): DataFrame with 'chromosome', 'start', 'end', and 'strand'.
#         maf_file_template (str): Path template for gzipped MAF files.
#         ref_species (str): Identifier for the reference species (e.g., "hg38").

#     Returns:
#         pd.Series: A Series of alignment dictionaries, indexed to match the input DataFrame.
#     """
#     results = [None] * len(exons_df)

#     # Group exons by chromosome to process one MAF file at a time
#     for chrom, group in exons_df.groupby('chromosome'):
#         print(f"🧬 Processing {len(group)} exons for chromosome: {chrom}")

#         # (AT)
#         chrom = "chr17"
#         maf_path = maf_file_template+f"{chrom}.maf.gz"
#         ref_prefix = f"{ref_species}.{chrom}"
        
#         # Prepare the list of exons for the worker function
#         exons_to_find = [{'idx': idx, **row.to_dict()} for idx, row in group.iterrows()]

#         # Call the worker to process the file and yield results
#         for original_idx, species_aligns in _process_single_maf(maf_path, exons_to_find, ref_prefix):
#             # Place the result in the correct position corresponding to the original DataFrame
#             original_loc = exons_df.index.get_loc(original_idx)
#             results[original_loc] = species_aligns
            
#     return pd.Series(results, index=exons_df.index)
