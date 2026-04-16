import pandas as pd
import os


def map_path_names(path):
    """
    Searches a directory for all BAM files in a given path. Constructs and returns a Pandas dataframe with these paths
    """
    bam_paths = []

    for dirname, _, filenames in os.walk(path):
        for f in filenames:
            # Check all BAM files. Exclude duplicates (called "Aligned.sortedByCoord.out.bam.CB.bam")
            if f.endswith(".bam") and "Aligned.sortedByCoord.out.bam.CB.bam" not in f:
                full_path = os.path.join(dirname, f)
                cell_id = f.split(".")[0]
                bam_paths.append((cell_id, full_path))
    
    bam_df = pd.DataFrame(bam_paths, columns=["clean_cell_id", "bam_path"])
    return bam_df

def safe_exists(path):
    """
    Safe implementation of os.path.exists to account for None values (ex. for TSP30)
    """
    if path:
        return os.path.exists(path)
    else:
        return False
    

    
if __name__ == "__main__":

    DATA_DIR = "/gpfs/commons/projects/knowles_singlecell_splicing/TabulaSenis/data/AWS/"
    tm = pd.read_csv(os.path.join(DATA_DIR, "metadata", "tabula-muris-senis-facs-official-raw-obj__cell-metadata__cleaned_ids.csv"))

    # Info about dataset
    print("----- TABULA MURIS SENIS METADATA -----\n")

    print(f"Total cells: {len(tm)}")
    print(f"Total specimens: {len(tm['mouse.id'].unique())}\n")

    print(f"Cell Types with over 30 cells:")
    counts = tm['cell_ontology_class'].value_counts()
    common_cell_types = counts[counts > 30]
    print(common_cell_types)


    print(f"\n----------------------------------------\n")


    print("Searching all BAM files in directory...\n")
    bam_df = map_path_names(os.path.join(DATA_DIR, "Plate_seq"))
    print(f"Total files found: {len(bam_df)}\n")

    # Check for duplicate cell IDs found
    dupes = bam_df["clean_cell_id"].value_counts()
    dupes = dupes[dupes > 1]

    if len(dupes) > 0:
        print("Duplicate BAMs found!")
        print(dupes.head(20))
        raise ValueError("Fix duplicates before continuing")

    # Merge tables and save new dataframe
    tm = tm.merge(
        bam_df,
        on="clean_cell_id",
        how="left"
    )

    print(f"Length of new dataframe after merge: {len(tm)}")

    tm.to_csv("/gpfs/commons/home/nkeung/tabula_muris_data/bam_paths.csv", index=False)
