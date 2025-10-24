import pickle
import numpy as np
import pandas as pd





# main_dir = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/'
# file_name = 'dummy_exon_intron_positions_shrt2_5and3prime_3primeIntronSeq.pkl'
# pickle_file_path = f'{main_dir}{file_name}'
# with open(pickle_file_path, 'rb') as f:
#     data = pickle.load(f)
# print(f"File loaded: {pickle_file_path}")
# print(f"Data type: {type(data)}")


# main_dir = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/'
# file_name = 'dummy_exon_intron_positions_shrt2_5and3prime_5primeIntronSeq.pkl'
# pickle_file_path = f'{main_dir}{file_name}'
# with open(pickle_file_path, 'rb') as f:
#     data = pickle.load(f)
# print(f"File loaded: {pickle_file_path}")
# print(f"Data type: {type(data)}")


# main_dir = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/'
# file_name = 'dummy_exon_intron_positions_shrt2_5and3prime_ExonSeq.pkl'
# pickle_file_path = f'{main_dir}{file_name}'
# with open(pickle_file_path, 'rb') as f:
#     data = pickle.load(f)

# print(f"File loaded: {pickle_file_path}")
# print(f"Data type: {type(data)}")



main_dir = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/'
file_name = 'test_3primeIntron_filtered.pkl'
pickle_file_path = f'{main_dir}{file_name}'
with open(pickle_file_path, 'rb') as f:
    data = pickle.load(f)

print(f"File loaded: {pickle_file_path}")
print(f"Data type: {type(data)}")


file_name = 'test_5primeIntron_filtered.pkl'
pickle_file_path = f'{main_dir}{file_name}'
with open(pickle_file_path, 'rb') as f:
    data = pickle.load(f)

print(f"File loaded: {pickle_file_path}")
print(f"Data type: {type(data)}")


file_name = 'test_ExonSeq_filtered_nonJoint.pkl'
pickle_file_path = f'{main_dir}{file_name}'
with open(pickle_file_path, 'rb') as f:
    data = pickle.load(f)

print(f"File loaded: {pickle_file_path}")
print(f"Data type: {type(data)}")


file_name = 'test_ExonSeq_filtered.pkl'
pickle_file_path = f'{main_dir}{file_name}'
with open(pickle_file_path, 'rb') as f:
    data = pickle.load(f)

print(f"File loaded: {pickle_file_path}")
print(f"Data type: {type(data)}")


file_name = 'test_merged_filtered_min30Views.pkl'
pickle_file_path = f'{main_dir}{file_name}'
with open(pickle_file_path, 'rb') as f:
    data = pickle.load(f)

print(f"File loaded: {pickle_file_path}")
print(f"Data type: {type(data)}")




