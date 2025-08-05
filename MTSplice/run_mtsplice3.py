from mmsplice.mtsplice import MTSplice
from mmsplice.utils import encodeDNA
import pandas as pd
import pickle


def get_windows_with_padding(full_seq, len_5p, len_exon, len_3p,
                             tissue_acceptor_intron, tissue_acceptor_exon,
                             tissue_donor_exon, tissue_donor_intron):
    # Acceptors: region around exon start (3' splice site)
    acceptor_intron = len_3p  # region before the exon (3' intron)
    # Donor: region around exon end (5' splice site)
    donor_intron = len_5p     # region after the exon (5' intron)

    # Get acceptor window
    acceptor_start = len_5p + 0 - tissue_acceptor_intron
    acceptor_end = len_5p + tissue_acceptor_exon

    # Pad acceptor if needed
    seq_acceptor = full_seq[max(0, acceptor_start):acceptor_end]
    if acceptor_start < 0:
        seq_acceptor = "N" * abs(acceptor_start) + seq_acceptor
    if len(seq_acceptor) < tissue_acceptor_intron + tissue_acceptor_exon:
        seq_acceptor = seq_acceptor + "N" * (tissue_acceptor_intron + tissue_acceptor_exon - len(seq_acceptor))

    # Get donor window
    donor_end = len_5p + len_exon + tissue_donor_intron
    donor_start = len_5p + len_exon - tissue_donor_exon

    seq_donor = full_seq[donor_start:donor_end]
    if donor_end > len(full_seq):
        seq_donor = seq_donor + "N" * (donor_end - len(full_seq))
    if donor_start < 0:
        seq_donor = "N" * abs(donor_start) + seq_donor
    if len(seq_donor) < tissue_donor_exon + tissue_donor_intron:
        seq_donor = seq_donor + "N" * (tissue_donor_exon + tissue_donor_intron - len(seq_donor))

    return {
        'acceptor': seq_acceptor,
        'donor': seq_donor
    }



# Retina index
retina_index = [0]
file = '/home/atalukder/Contrastive_Learning/data/final_data/ASCOT_finetuning/psi_variable_Retina___Eye_psi_MERGED.pkl'
# Replace with your actual file path
with open(file, "rb") as f:
    data = pickle.load(f)


mtsplice = MTSplice(deep=True)

retina_predictions = {}


acceptor_seqs = []
donor_seqs = []
exon_ids = []

for exon_id, info in data.items():
    intron_5p = info['5p']
    exon_seq = info['exon']['start'] + info['exon']['end']
    intron_3p = info['3p']
    # full_seq = intron_5p + exon_seq + intron_3p
    full_seq = intron_5p + exon_seq + intron_3p

    len_5p = len(intron_5p)
    len_exon = len(exon_seq)
    len_3p = len(intron_3p)

    
    out = get_windows_with_padding(
        full_seq, len_5p, len_exon, len_3p,
        tissue_acceptor_intron=300, tissue_acceptor_exon=100,
        tissue_donor_exon=100, tissue_donor_intron=300
    )
    acceptor_seqs.append(out['acceptor'])
    donor_seqs.append(out['donor'])
    exon_ids.append(exon_id)  

# batch = {'acceptor': acceptor_seqs, 'donor': donor_seqs}
batch = mtsplice.prepare_batch(acceptor_seqs, donor_seqs)
result = mtsplice.predict_on_batch(batch)

import pandas as pd
import numpy as np
from scipy.special import logit, expit
from scipy.stats import spearmanr

########################## psi splicing ##########################

retina_pred = result[:, 10]    # First column = Retina - Eye logit(delta)

# Build predictions DataFrame
predictions = pd.DataFrame({
    'exon_id': exon_ids,
    'Retina - Eye': retina_pred
})


ground_truth = pd.read_csv("/home/atalukder/Contrastive_Learning/data/ASCOT/variable_cassette_exons_with_logit_mean_psi.csv")

tissue = "Retina - Eye"

# ---- Merge on exon_id ----
df = pd.merge(
    ground_truth[['exon_id', 'logit_mean_psi', tissue]],  # ground truth PSI
    predictions[['exon_id', tissue]],   # predicted logit(delta)
    on='exon_id',
    suffixes=('_true', '_logit_delta_pred')
)


# ---- Compute final predicted PSI: sigmoid(logit_delta + logit_mean_psi) ----
df['final_predicted_psi'] = expit(df['Retina - Eye_logit_delta_pred'] + df['logit_mean_psi'])   # as percent

# ---- Spearman in PSI space ----
valid = (df[f'{tissue}_true'] >= 0) & (df[f'{tissue}_true'] <= 100) & (~df[f'{tissue}_true'].isnull())
rho_psi, _ = spearmanr(df.loc[valid, f'{tissue}_true'], df.loc[valid, 'final_predicted_psi'])
print(f"\n🔬 Spearman ρ (PSI): {rho_psi:.4f}")

df.to_csv(
    f"tsplice_final_predictions_{tissue.replace(' ', '_')}.tsv", sep="\t", index=False
)
print(df.head())



########################## differential psi splicing ##########################

ground_truth = pd.read_csv("/home/atalukder/Contrastive_Learning/data/ASCOT/variable_cassette_exons_with_logit_mean_psi.csv")

tissue = "Retina - Eye"

# ---- Merge on exon_id ----
df = pd.merge(
    ground_truth[['exon_id', 'logit_mean_psi', tissue]],  # ground truth PSI
    predictions[['exon_id', tissue]],   # predicted logit(delta)
    on='exon_id',
    suffixes=('_true', '_logit_delta_pred')
)


# 1. Calculate ground-truth logit-delta for each exon
eps = 1e-6
df['Retina_frac_true'] = np.clip(df[f'{tissue}_true'] / 100, eps, 1 - eps)
df['truth_delta_psi'] = logit(df['Retina_frac_true']) - df['logit_mean_psi']

# 2. Prepare valid mask (avoid -1 and NaN in both columns)
valid = (
    (df[f'{tissue}_true'] >= 0) & (df[f'{tissue}_true'] <= 100) & (~df[f'{tissue}_true'].isnull()) &
    (~df['logit_mean_psi'].isnull()) & (~df[f'{tissue}_logit_delta_pred'].isnull())
)

# 3. Compute Spearman correlation between truth_delta_psi and predicted logit-delta
rho_delta, _ = spearmanr(
    df.loc[valid, 'truth_delta_psi'],
    df.loc[valid, f'{tissue}_logit_delta_pred']
)
print(f"\n🔬 Spearman ρ (logit-delta): {rho_delta:.4f}")