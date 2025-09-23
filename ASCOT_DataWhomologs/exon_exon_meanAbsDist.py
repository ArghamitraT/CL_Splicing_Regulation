
from utils import load_csv, get_tissue_PSI_ASCOT, save_matrix, compute_meanAbsoluteDistance, compute_meanAbsoluteDistance_blockwise
import time

def main():
    start = time.time()
    # File paths
    file_name = "train"  # "train", "val", "test", "variable"
    file_ascot = f'/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/{file_name}_cassette_exons.csv'
    file_path = file_ascot
    output_path = f"/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/{file_name}_ExonExon_meanAbsDist.pkl"

    Multiz_overlap_file = f"/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/{file_name}_cassette_exons_multizOverlaps.csv"
    
    # Load exon file
    df = load_csv(file_path)

    # Get tissue expression matrix
    expr_matrix = get_tissue_PSI_ASCOT(df)

    exon_ids = df["exon_id"]

    # if file_name == 'train':
    #     mad_df = compute_meanAbsoluteDistance_blockwise(expr_matrix, exon_ids)
    # else:   
    #     mad_df = compute_meanAbsoluteDistance(expr_matrix, exon_ids)

    mapping_df = load_csv(Multiz_overlap_file)  # the file you pasted

    # Build mapping dict: ascot_exon_id → Exon Name
    id_to_name = dict(zip(mapping_df["ascot_exon_id"], mapping_df["Exon Name"]))

    # Replace exon_ids (which are ascot_exon_id) with Exon Name
    exon_ids_named = exon_ids.map(id_to_name).fillna(exon_ids)  # fallback to original if not found

    # Now call your function
    if file_name == 'train':
        mad_df = compute_meanAbsoluteDistance_blockwise(expr_matrix, exon_ids_named)
    else:   
        mad_df = compute_meanAbsoluteDistance(expr_matrix, exon_ids_named)

    # Save
    save_matrix(mad_df, output_path)
    print(f"✅ Correlation matrix saved to {output_path}")

    end = time.time()
    print(f"✅ Total runtime: {end - start:.2f} seconds")



if __name__ == "__main__":
    main()
