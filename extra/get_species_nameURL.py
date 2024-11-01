"""
This file was used to get the scientific names of all speces. like human hg38 and gorilla it is gorGor3
"""

import re
import csv

# Define the input file path and output CSV file
input_file = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/species_name.txt'  # Replace with the actual path to your file
output_csv = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/species_refSeq_urls2.csv'

# Define base URL for constructing the fasta download links
base_url = "http://hgdownload.soe.ucsc.edu/goldenPath/{species}/bigZips/{species}.fa.gz"

# Function to extract the species name from a line
def extract_species_name(line):
    # Use regular expressions to extract the species code (e.g., 'punNye1' from 'PunNye1.0/punNye1')
    match = re.search(r"/([a-zA-Z]+[0-9]*)\s", line)
    return match.group(1) if match else None

def extract_species_info(line):
    # Use regex to capture both the common name and the species code
    common_name = line.split()[0]

    # Use regex to extract the species code (e.g., 'panTro4')
    match = re.search(r"/([a-zA-Z]+[0-9]*)\s", line)
    species_code = match.group(1) if match else None
    
    return common_name, species_code


# Open the output CSV file for writing
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write the header row
    writer.writerow(["Species Name", "URL", "Common Name"])
    
    # Read the species names from the file and construct URLs
    with open(input_file, 'r') as file:
        for line in file:
            # species_name = extract_species_name(line.strip())
            common_name, species_name = extract_species_info(line.strip())
            
            if species_name:
                # Construct the download URL
                download_url = base_url.format(species=species_name)
                
                # Write the species name and URL to the CSV file
                writer.writerow([species_name, download_url, common_name])

print(f"URLs saved successfully in {output_csv}.")
