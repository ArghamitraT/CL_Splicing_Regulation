import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cell_type = "pericyte"
DIR = f"/gpfs/commons/home/nkeung/tabula_sapiens/{cell_type}/output"

df = pd.read_csv(f"{DIR}/SE.MATS.JCEC.txt", sep="\t")

def total_count(s):
    return sum(float(x) for x in s.split(',') if x != 'NA')

def per_replicate_psi(row):
    ijc_vals = row['IJC_SAMPLE_1'].split(',')
    sjc_vals = row['SJC_SAMPLE_1'].split(',')
    psi_list = []
    
    for i, s in zip(ijc_vals, sjc_vals):
        if i == 'NA' or s == 'NA':
            psi_list.append(np.nan)
            continue
        
        e = row['exonEnd'] - row['exonStart_0base']
        len_i = 99 + min(e, 99) + max(0, e - 100 + 1)
        len_s = 99
        i_norm = float(i) / len_i
        s_norm = float(s) / len_s
        denom = i_norm + s_norm
        psi_list.append(np.nan if denom == 0 else i_norm / denom)
    
    # Return as comma-separated string or list, depending on what you need
    return ','.join(f'{p:.3f}' if not np.isnan(p) else 'NA' for p in psi_list)

def check_match(row):
    psi_vals = row['PSI_per_replicate'].split(',')
    inc_vals = row['IncLevel1'].split(',')
    
    # Must have same number of replicates
    if len(psi_vals) != len(inc_vals):
        return False
    
    for psi, inc in zip(psi_vals, inc_vals):
        if psi == 'NA' and inc == 'NA':
            continue
        if psi == 'NA' or inc == 'NA':
            return False
        
        # Convert to float and allow for small rounding tolerance
        if not np.isclose(float(psi), float(inc), atol=1e-3):
            return False
    return True

df['PSI_per_replicate'] = df.apply(per_replicate_psi, axis=1)

# Apply across rows and stop at the first mismatch
for idx, row in df.iterrows():
    if not check_match(row):
        print(f"❌ Mismatch found at row {idx}")
        print("PSI_per_replicate:", row['PSI_per_replicate'])
        print("IncLevel1:", row['IncLevel1'])
        break
else:
    print("✅ All PSI_per_replicate values match IncLevel1")
    
df['inclusion_count'] = df['IJC_SAMPLE_1'].apply(total_count)
df['exclusion_count'] = df['SJC_SAMPLE_1'].apply(total_count)
df['PSI'] = df.apply(
    lambda row: row['inclusion_count'] / (row['inclusion_count'] + row['exclusion_count'])
    if (row['inclusion_count'] + row['exclusion_count']) > 0 else np.nan,
    axis=1
)

# Show PSI distribution
# psi_list = df['PSI'].tolist()
# plt.figure(figsize=(10,6))
# plt.hist(psi_list, bins=np.arange(0, 1.1, 0.1))
# plt.xlabel("Mean PSI")
# plt.ylabel("Number of Events")
# plt.title("PSI Distribution for A3SS Event")
# plt.tight_layout()
# plt.savefig("/gpfs/commons/home/nkeung/Contrastive_Learning/code/figures/psi_a3ss.jpg", dpi=300, bbox_inches="tight")
