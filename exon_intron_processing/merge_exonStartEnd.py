import pickle

file_name = 'train'
# Load the pickle file
# pickle_file_path = f'/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data_new/intronExonSeq_multizAlignment_noDash/trainTestVal_data/{file_name}_ASCOT_ExonSeq.pkl'
pickle_file_path = f'/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data_new/intronExonSeq_multizAlignment_noDash/trainTestVal_data/{file_name}_ExonSeq_filtered_nonJoint.pkl'

with open(pickle_file_path, 'rb') as f:
    msa_results_list_unTOKEN = pickle.load(f)

# Function to concatenate padded start + end
def _process_exon(exon_dict):
    start = exon_dict.get("start", "")
    end = exon_dict.get("end", "")
    start_padded = start.ljust(100, "N")
    end_padded = end.rjust(100, "N")
    return start_padded + end_padded

# New dictionary to hold the processed output
processed_sequences = {}

for exon_id, exon_data in msa_results_list_unTOKEN.items():
    exon_start = exon_data.get("exon_start", {})
    exon_end = exon_data.get("exon_end", {})
    
    # Only process if both start and end have the same species set
    if set(exon_start.keys()) == set(exon_end.keys()) and len(exon_start) >= 2:
        processed_sequences[exon_id] = {}
        for species in exon_start:
            processed_sequences[exon_id][species] = _process_exon({
                "start": exon_start[species],
                "end": exon_end[species]
            })

# Save to new pickle file
output_path = f'/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data_new/intronExonSeq_multizAlignment_noDash/trainTestVal_data/{file_name}_ExonSeq_filtered.pkl'

with open(output_path, 'wb') as f:
    pickle.dump(processed_sequences, f)

print(f"Saved {len(processed_sequences)} processed exons to: {output_path}")
