import pickle


file_name = 'train_ExonSeq'
# Load original .pkl file
with open(f'/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data_unfiltered/{file_name}.pkl', 'rb') as f:
    msa_results_list_unTOKEN = pickle.load(f)

# Filter: keep entries with ≥2 species in either exon_start or exon_end
filtered = {
    k: v for k, v in msa_results_list_unTOKEN.items()
    if len(v.get('exon_start', {})) == len(v.get('exon_end', {})) >= 2
}

# Save filtered result to a new .pkl file
with open(f'/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/{file_name}_filtered.pkl', 'wb') as f:
    pickle.dump(filtered, f)

print(f"Filtered dataset saved with {len(filtered)} entries.")
