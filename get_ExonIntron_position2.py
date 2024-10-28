import pandas as pd
import re
import pickle

# Define the input file path
input_file = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/knownGene.multiz100way.exonNuc.fa'  # Replace with the path to your .fa file
# input_file = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/dummy.fa'  # Replace with the path to your .fa file

# List to store the extracted data
data = []
# 
# Regular expression to match each line and extract required details
# pattern = re.compile(r"^(\S+) (\d+) \d+ \d+ (chr\d+):(\d+)-(\d+)([+-]?)")
pattern = re.compile(r"^(\S+) (\d+) \d+ \d+ (\S+):(\d+)-(\d+)([+-]?)")

# Function to parse the exon name
def get_exon_name(identifier):
    # Example: "ENST00000684719.1_hg38_1_6" -> "ENST00000684719.1_1_6"
    parts = identifier.split('_')
    return f"{parts[0][1:]}_{parts[2]}_{parts[3]}"

# Function to determine intron start and end based on strand
def calculate_intron(start, end, strand):
    if strand == '+':
        # Check if the exon start is large enough to calculate upstream
        if start > 200:
            intron_start = start - 200
            intron_end = start - 1
        else:
            intron_start, intron_end = None, None  # Invalid upstream case
    else:
        # For the negative strand, calculate the downstream (as upstream is after the exon end)
        intron_start = end + 1
        intron_end = end + 200

    # Ensure the results are either valid integers or None
    return intron_start, intron_end

# Read the file and extract information
with open(input_file, 'r') as file:
    for line in file:
        line = line.strip()
        match = pattern.match(line)
        if match:
            identifier = match.group(1)
            length = match.group(2)  # Could be used if needed
            chromosome = match.group(3)
            exon_start = int(match.group(4))
            exon_end = int(match.group(5))
            strand = match.group(6) if match.group(6) else 'Unknown'

            # Extract species, chromosome, and position
            parts = identifier.split('_')
            species_name = parts[1]  # e.g., "hg38"
            exon_name = get_exon_name(identifier)
            
            # Calculate intron start and end
            intron_start, intron_end = calculate_intron(exon_start, exon_end, strand)

            # Save the data
            data.append([exon_name, species_name, chromosome, exon_start, exon_end, strand, intron_start, intron_end])

# Create a DataFrame
df = pd.DataFrame(data, columns=[
    'Exon Name', 'Species Name', 'Chromosome', 
    'Exon Start', 'Exon End', 'Strand', 
    'Intron Start', 'Intron End'
])

# Save the DataFrame to a CSV file
output_name = (input_file.split('/')[-1]).rsplit('.',1)[0]
final_name = f'/gpfs/commons/home/atalukder/Contrastive_Learning/data/multiz100way/alignment/{output_name}_exon_intron_positions'

df.to_pickle(final_name+'.pkl')
df.to_csv(final_name+'.csv', index=False)



print("Exon and intron details have been successfully saved to 'exon_intron_details.csv'.")
