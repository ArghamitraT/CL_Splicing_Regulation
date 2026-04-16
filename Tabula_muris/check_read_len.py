"""
Checks the sequence length of the first line in each BAM file for every BAM file in the dataset.
"""

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

df = pd.read_csv("/gpfs/commons/home/nkeung/tabula_sapiens/bam_paths.tsv", sep="\t")
df = df[df["path_exists"]]

paths = df["bam_path"].tolist()

def get_read_length(path):
    try:
        cmd = f"samtools view {path} | head -n 1 | awk '{{print length($10)}}'"
        length = int(subprocess.check_output(cmd, shell=True, executable="/bin/bash").decode().strip())
        return length
    except subprocess.CalledProcessError:
        return None

read_lengths = set()
terminate_flag = False

with ThreadPoolExecutor(max_workers=6) as executor:
    future_to_bam = {executor.submit(get_read_length, bam): bam for bam in paths}
    for future in as_completed(future_to_bam):
        length = future.result()
        if length is not None:
            read_lengths.add(length)
            if length != 100:
                print(f"Variable read length detected in {future_to_bam[future]} -> {length}")
                break

# Optional: check if all are 100
if read_lengths == {100}:
    print(f"✅ First lengths of all files is 100")
else:
    print(f"⚠️ Unexpected read length found: {read_lengths}")
