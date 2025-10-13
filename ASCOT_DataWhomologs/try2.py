import pickle


division = "val"
weight_matrix_path = f"/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/ASCOT_data/{division}_ExonExon_meanAbsDist_ASCOTname.pkl"
with open(weight_matrix_path, "rb") as f:
    weight_matrix_df1 = pickle.load(f)



weight_matrix_path2 = f"/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/ASCOT_mad_permultiz_temp/{division}_meanAbsDist_perMultiz.pkl"
with open(weight_matrix_path2, "rb") as f:
    weight_matrix_df2 = pickle.load(f)


print()