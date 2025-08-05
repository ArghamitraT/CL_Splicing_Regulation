

input_gtf = "/home/atalukder/Contrastive_Learning/data/MTSplice/variable_cassette_exons.gtf"
output_gtf = "/home/atalukder/Contrastive_Learning/data/MTSplice/variable_cassette_exons_genename.gtf"

with open(input_gtf) as fin, open(output_gtf, "w") as fout:
    for line in fin:
        if line.strip() == "" or line.startswith("#"):
            fout.write(line)
            continue
        fields = line.strip().split('\t')
        # attributes is the last field
        attributes = fields[-1]
        # Extract gene_id value
        import re
        m = re.search(r'gene_id "([^"]+)"', attributes)
        if m:
            gene_id = m.group(1)
            # Check if gene_name already exists
            if 'gene_name' not in attributes:
                # Append gene_name at the end
                attributes = attributes.strip() + f' gene_name "{gene_id}";'
        fields[-1] = attributes
        fout.write('\t'.join(fields) + '\n')



# import pickle

# file = '/home/atalukder/Contrastive_Learning/data/final_data/ASCOT_finetuning/psi_variable_Retina___Eye_psi_MERGED.pkl'
# # Replace with your actual file path
# with open(file, "rb") as f:
#     data = pickle.load(f)

# print(type(data))
# print(data)
