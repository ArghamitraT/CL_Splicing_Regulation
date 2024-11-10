import pickle
import os
import math
import argparse

def remove_exons_with_few_introns(merged_data):
    """
    Identifies and removes exons with 0 or 1 intron from the data.

    Args:
        merged_data (dict): Merged data of exons and their introns.

    Returns:
        dict: Data with exons containing more than 1 intron.
    """
    filtered_data = {
        exon: introns
        for exon, introns in merged_data.items()
        if len(introns) > 1
    }
    return filtered_data

def filter_intronic_sequences(introns, max_n_percentage=0.05):
    """
    Filters intronic sequences to remove those with more than max_n_percentage of 'N'.
    
    Args:
        introns (list): List of intronic sequences.
        max_n_percentage (float): Maximum allowed percentage of 'N' in a sequence.
        
    Returns:
        list: Filtered list of intronic sequences.
    """
    filtered_introns = {}
    for species, sequence in introns.items():
        n_count = sequence.count('N')
        n_percentage = n_count / len(sequence) if len(sequence) > 0 else 0
        if n_percentage <= max_n_percentage:
            filtered_introns[species] = sequence
    return filtered_introns

def merge_and_save_exon_data(data_dir, file_names, max_n_percentage=0.05):
    merged_data = {}
    all_exon_names = set()

    # Load each file, filter introns, and merge contents
    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            # Filter introns in each exon
            data = {exon: filter_intronic_sequences(introns, max_n_percentage) for exon, introns in data.items()}
            merged_data.update(data)

    # Remove exons with 0 or 1 intron after filtering
    merged_data = remove_exons_with_few_introns(merged_data)
    all_exon_names.update(merged_data.keys())

    # Save the merged data into a new .pkl file
    output_pkl_path = os.path.join(data_dir, 'merged_intron_sequences.pkl')
    with open(output_pkl_path, 'wb') as output_file:
        pickle.dump(merged_data, output_file)

    print("Merging complete. The merged data is saved to 'merged_intron_sequences.pkl'")
    return output_pkl_path, all_exon_names

def generate_all_exon_names(output_txt_filepath, exon_names):
    with open(output_txt_filepath, 'w') as txt_file:
        for exon_name in sorted(exon_names):
            txt_file.write(f"{exon_name}\n")
    print(f"Exon names saved to {output_txt_filepath}")

def main(data_dir, file_names, max_n_percentage):
    # Step 1: Merge all files, filter intronic sequences, and remove exons with 0 or 1 intron
    output_pkl_path, final_exon_names = merge_and_save_exon_data(data_dir, file_names, max_n_percentage)

    # Step 2: Generate text file with all exon names in the final merged .pkl file
    output_txt_path = os.path.join(data_dir, 'all_exon_names.txt')
    generate_all_exon_names(output_txt_path, final_exon_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process exon and intron data files.")
    parser.add_argument("data_dir", type=str, help="Directory containing the data files")
    parser.add_argument("file_names", type=str, nargs='+', help="List of data file names")
    parser.add_argument("--max_n_percentage", type=float, default=0.05, help="Maximum allowed percentage of 'N' in a sequence")

    args = parser.parse_args()
    main(args.data_dir, args.file_names, args.max_n_percentage)