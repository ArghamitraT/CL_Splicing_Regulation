"""
Downloads the annotation files
"""

import csv
import os
import requests
from tqdm import tqdm

# Path to the CSV file
csv_file_path = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/species_refSeq_urls.csv'  # Replace with your actual file path
output_main_dir = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/gene_annotation/raw_data'
# Base URL template
base_url_template = "http://hgdownload.soe.ucsc.edu/goldenPath/{species}/database/"

# File types to download
file_types = ["refGene.txt.gz", "refGene.sql"]

# Dictionary to track available species and their file status
download_report = {}

# Function to download file with renaming
def download_file(url, output_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192), desc=f"Downloading {output_path}", unit='KB'):
                f.write(chunk)
        return True
    except requests.exceptions.HTTPError:
        print(f"File not found: {url}")
        return False

# Reading species from the CSV and downloading files
with open(csv_file_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header if present
    
    for row in reader:
        species_code = row[0].strip()
        found_files = 0
        
        for file_type in file_types:
            url = f"{base_url_template.format(species=species_code)}{file_type}"
            # Handling renaming based on file extension
            if file_type == "refGene.sql":
                renamed_file = f"refGene_{species_code}.sql"
            elif file_type == "refGene.txt.gz":
                renamed_file = f"refGene_{species_code}.txt.gz"
           
            # renamed_file = f"{file_type.split('.')[0]}_{species_code}.{file_type.split('.')[1]}"
            final_saved_path=os.path.join(output_main_dir, renamed_file)
            
            # Download and rename file
            if download_file(url, final_saved_path):
                found_files += 1
        
        # Update download report
        download_report[species_code] = found_files

# Reporting
total_species = len(download_report)
available_species = {k: v for k, v in download_report.items() if v > 0}

print(f"Total species checked: {total_species}")
print(f"Species with available files: {len(available_species)}")
print("Details of available species and the number of files found:")

for species, files_found in available_species.items():
    print(f"{species}: {files_found} files found")
