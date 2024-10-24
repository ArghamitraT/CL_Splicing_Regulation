import pickle
import numpy as np
import os
import argparse

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    print("Data loaded successfully.")
    return np.array(data)

def process_introns(data):
    num_introns = data.shape[0]
    print(f"Processing {num_introns} introns...")

    for intron_idx in range(num_introns):
        print(f"Processing intron {intron_idx + 1}/{num_introns}...")
        
        # Extract data for one intron (shape: (seq_len, species))
        intron_matrix = data[intron_idx, :, :].copy()
        
        # Convert byte characters to strings
        intron_matrix = np.char.decode(intron_matrix.astype(np.bytes_), 'utf-8')
        
        intron_matrix = intron_matrix.T

        print(f"Intron {intron_idx + 1} processed.")
        yield intron_idx, intron_matrix

def save_intron_matrices(data, output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")
        
    for intron_idx, intron_matrix in process_introns(data):
        # Create a filename for each intron
        filename = f"{output_dir}/intron_{intron_idx}_matrix.npy"
        
        # Save the matrix as a .npy file
        np.save(filename, intron_matrix)
        
        print(f"Saved {filename} with shape {intron_matrix.shape}")

def main(file_path, output_dir):
    # Load, process, and save the data
    data = load_data(file_path)
    save_intron_matrices(data, output_dir)
    print("All introns processed and saved successfully.")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process and save intron matrices from a pickle file.")
    
    # Add arguments for file_path and output_dir
    parser.add_argument('file_path', type=str, help="Path to the input pickle file.")
    parser.add_argument('output_dir', type=str, help="Directory to save the processed intron matrices.")
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Call main function with parsed arguments
    main(args.file_path, args.output_dir)
