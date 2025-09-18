import pickle
import pandas as pd



def load_pkl(file_path: str) -> pd.DataFrame:
    """
    Load pickled DataFrame.
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    # mad_df = pd.read_pickle(file_path)
    return mad_df


division = "val"
file_path = f"/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/{division}_3primeIntron_filtered.pkl"
exon_list = f"/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/{division}_exon_list.csv"

df = load_pkl(file_path)
# print(df.head())


# 1. Get the keys (exon names) from the dictionary and convert to a list
exon_names = list(df.keys())

# 2. Create a pandas DataFrame from the list
#    Specify the column name directly during creation
exons_df = pd.DataFrame(exon_names, columns=['exon_id'])

# 3. Save the DataFrame to a CSV file
#    Use index=False to avoid writing the DataFrame index as a column
exons_df.to_csv(exon_list, index=False)
print(f"Successfully saved exon IDs to '{exon_list}'")
print("\nFirst 5 rows of the new DataFrame:")
print(exons_df.head())
