import csv
import re
import os
import glob

# Directory containing the SQL and TXT files
data_dir = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/gene_annotation/'  # Replace with the path to your folder
output_dir = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/gene_annotation/gene_annotation_csv/'  # Directory to save the generated CSV files
os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

# Columns to be extracted
columns_to_extract = ["exonCount", "exonStarts", "exonEnds", "chrom", "strand", "name"]

# Function to parse SQL file and get exact column indices
# Function to parse SQL file and get exact column indices, ignoring lines with 'KEY'
def parse_sql_file(sql_file_path):
    column_indices = {}
    inside_create_table = False
    current_index = 0
    
    with open(sql_file_path, 'r') as sql_file:
        for line in sql_file:
            # Check if we are inside the CREATE TABLE section
            if "CREATE TABLE" in line:
                inside_create_table = True
                continue
            if inside_create_table:
                # Stop when we reach the end of the table definition
                if line.strip().startswith(")"):
                    break
                # Skip lines that define keys
                if line.strip().startswith("KEY"):
                    continue
                
                # Match column definitions and track the index
                match = re.search(r"`(\w+)`", line)
                if match:
                    column_name = match.group(1)
                    # print(f"Column '{column_name}' found at index {current_index}")
                    if column_name in columns_to_extract:
                        column_indices[column_name] = current_index
                    current_index += 1
    return column_indices

# Function to parse TXT file and extract exon data
def parse_txt_file(txt_file_path, column_indices, species_name, output_csv_path):
    with open(txt_file_path, 'r') as txt_file, open(output_csv_path, 'w') as csv_file:
        # Adding "chromosome" and "strand" to the output columns
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["species_name", "chromosome", "strand", "exon_name", "exon_start", "exon_end"])

        for line in txt_file:
            data = line.strip().split("\t")

            # Extract required fields based on indices found from SQL file
            transcript_name = data[column_indices["name"]]
            chromosome = data[column_indices["chrom"]]
            strand = data[column_indices["strand"]]
            exon_count = int(data[column_indices["exonCount"]])
            exon_starts = list(map(int, data[column_indices["exonStarts"]].strip(',').split(',')))
            exon_ends = list(map(int, data[column_indices["exonEnds"]].strip(',').split(',')))
            
            # Generate rows for each exon
            for i in range(exon_count):
                exon_name = f"{transcript_name}_{i+1}"
                exon_start = exon_starts[i]
                exon_end = exon_ends[i]
                
                # Write the row with species name, chromosome, strand, exon details
                csv_writer.writerow([species_name, chromosome, strand, exon_name, exon_start, exon_end])


# Process each pair of SQL and TXT files in the directory
for sql_file_path in glob.glob(os.path.join(data_dir, "*.sql")):
    print(sql_file_path)
    # Extract species name from the SQL file name
    species_name = re.search(r'refGene_(.+)\.sql', os.path.basename(sql_file_path)).group(1)
    
    # Corresponding TXT file path
    txt_file_path = os.path.join(data_dir, f"refGene_{species_name}.txt")
    if not os.path.exists(txt_file_path):
        print(f"TXT file not found for species: {species_name}")
        continue

    # Parse the SQL file to get column indices
    column_indices = parse_sql_file(sql_file_path)

    # Output CSV file path
    output_csv_path = os.path.join(output_dir, f"{species_name}_exon_data.csv")

    # Parse the TXT file and write to CSV
    parse_txt_file(txt_file_path, column_indices, species_name, output_csv_path)

    print(f"CSV file created for species: {species_name}")
