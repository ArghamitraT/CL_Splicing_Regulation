from mmsplice.mtsplice import MTSplice
from mmsplice.utils import encodeDNA
import pandas as pd
import pickle

# Retina index
retina_index = [0]
file = '/home/atalukder/Contrastive_Learning/data/final_data/ASCOT_finetuning/psi_variable_Retina___Eye_psi_MERGED.pkl'
# Replace with your actual file path
with open(file, "rb") as f:
    data = pickle.load(f)

# data = dict(list(data.items())[:5]) ##(AT) comment


mtsplice = MTSplice(deep=True)

retina_predictions = {}

for exon_id, info in data.items():
    intron_3p = info['3p']
    intron_5p = info['5p']
    exon_seq = info['exon']['start'] + info['exon']['end']

    full_seq = intron_5p + exon_seq + intron_3p 
    overhang = (len(intron_3p), len(intron_5p))

    logit_psi = mtsplice.predict(full_seq, overhang=overhang)
    
    retina_predictions[exon_id] = logit_psi[0, retina_index]  # scalar

# Convert to DataFrame (optional)
retina_df = pd.DataFrame.from_dict(retina_predictions, orient='index', columns=['Retina_logit_psi'])



from scipy.special import logit
from scipy.stats import spearmanr
import numpy as np

# Step 1: Extract matching ground-truth psi values
retina_df["psi_val"] = retina_df.index.map(lambda k: data[k]["psi_val"])

# Step 2: Convert both to numpy arrays
y_pred_logit = retina_df["Retina_logit_psi"].values
y_true_raw = retina_df["psi_val"].values  # Assumes in percent [0–100]

# Step 3: Convert PSI to logit(ψ), with clipping to avoid logit(0) or logit(1)
eps = 1e-6
y_true_logit = logit(np.clip(y_true_raw / 100, eps, 1 - eps))

# Step 4: Compute Spearman correlation
rho, _ = spearmanr(y_true_logit, y_pred_logit)

print(f"\n🔬 Spearman ρ (Retina logit PSI): {rho:.4f}")


retina_df.to_csv("mtsplice_retina_predictions.tsv", sep="\t")



