from mmsplice.mtsplice import MTSplice
from mmsplice.utils import encodeDNA
import pandas as pd
import pickle


# def get_windows_with_padding(full_seq, len_5p, len_exon, len_3p,
#                              tissue_acceptor_intron, tissue_acceptor_exon,
#                              tissue_donor_exon, tissue_donor_intron):
#     # Acceptors: region around exon start (3' splice site)
#     acceptor_intron = len_3p  # region before the exon (3' intron)
#     # Donor: region around exon end (5' splice site)
#     donor_intron = len_5p     # region after the exon (5' intron)

#     # Get acceptor window
#     acceptor_start = len_5p + 0 - tissue_acceptor_intron
#     acceptor_end = len_5p + tissue_acceptor_exon

#     # Pad acceptor if needed
#     seq_acceptor = full_seq[max(0, acceptor_start):acceptor_end]
#     if acceptor_start < 0:
#         seq_acceptor = "N" * abs(acceptor_start) + seq_acceptor
#     if len(seq_acceptor) < tissue_acceptor_intron + tissue_acceptor_exon:
#         seq_acceptor = seq_acceptor + "N" * (tissue_acceptor_intron + tissue_acceptor_exon - len(seq_acceptor))

#     # Get donor window
#     donor_end = len_5p + len_exon + tissue_donor_intron
#     donor_start = len_5p + len_exon - tissue_donor_exon

#     seq_donor = full_seq[donor_start:donor_end]
#     if donor_end > len(full_seq):
#         seq_donor = seq_donor + "N" * (donor_end - len(full_seq))
#     if donor_start < 0:
#         seq_donor = "N" * abs(donor_start) + seq_donor
#     if len(seq_donor) < tissue_donor_exon + tissue_donor_intron:
#         seq_donor = seq_donor + "N" * (tissue_donor_exon + tissue_donor_intron - len(seq_donor))

#     return {
#         'acceptor': seq_acceptor,
#         'donor': seq_donor
#     }


def get_windows_with_padding(seq, overhang, tissue_acceptor_intron=300, tissue_acceptor_exon=100,
                             tissue_donor_exon=100, tissue_donor_intron=300):
            """
            Split seq for tissue specific predictions
            Args:
            seq: seqeunce to split
            overhang: (intron_length acceptor side, intron_length donor side) of
                        the input sequence
            """

            (acceptor_intron, donor_intron) = overhang

            assert acceptor_intron <= len(seq), "Input sequence acceptor intron" \
                " length cannot be longer than the input sequence"
            assert donor_intron <= len(seq), "Input sequence donor intron length" \
                " cannot be longer than the input sequence"

            # need to pad N if seq not enough long
            diff_acceptor = acceptor_intron - tissue_acceptor_intron
            if diff_acceptor < 0:
                seq = "N" * abs(diff_acceptor) + seq
            elif diff_acceptor > 0:
                seq = seq[diff_acceptor:]

            diff_donor = donor_intron - tissue_donor_intron
            if diff_donor < 0:
                seq = seq + "N" * abs(diff_donor)
            elif diff_donor > 0:
                seq = seq[:-diff_donor]

            return {
                'acceptor': seq[:tissue_acceptor_intron
                                + tissue_acceptor_exon],
                'donor': seq[-tissue_donor_exon
                            - tissue_donor_intron:]
            }


# Retina index
split_name = "test"
file = f'/home/atalukder/Contrastive_Learning/data/final_data/ASCOT_finetuning/psi_{split_name}_Retina___Eye_psi_MERGED.pkl'
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

    
    # out = get_windows_with_padding(
    #     full_seq, len_5p, len_exon, len_3p,
    #     tissue_acceptor_intron=300, tissue_acceptor_exon=100,
    #     tissue_donor_exon=100, tissue_donor_intron=300
    # )
    out = get_windows_with_padding(
        full_seq, overhang=(len_3p, len_5p), tissue_acceptor_intron=300, tissue_acceptor_exon=100,
                             tissue_donor_exon=100, tissue_donor_intron=300)
    
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

# --- 1. SETUP and DATA LOADING ---

# Assume these variables are already defined in your environment:
# result: np.array of shape (1181, 56) from your model
# exon_ids: list or array of exon IDs corresponding to the rows of 'result'
# split_name: e.g., 'validation' or 'test'

# Load ground truth data
ground_truth_path = f"/home/atalukder/Contrastive_Learning/data/ASCOT/{split_name}_cassette_exons_with_logit_mean_psi.csv"
ground_truth = pd.read_csv(ground_truth_path)

# --- 2. IDENTIFY TISSUE COLUMNS ---

# Programmatically get the list of 56 tissue names from the ground_truth columns
# Based on your screenshot, tissues start at column index 9 and end before the last 2 columns.
cols = list(ground_truth.columns)
# Find tissue columns
start_idx = cols.index('exon_boundary') + 1 if 'exon_boundary' in cols else 0
if 'chromosome' in cols:
    end_idx = cols.index('chromosome')
elif 'mean_psi' in cols:
    end_idx = cols.index('mean_psi')
else:
    end_idx = cols.index('logit_mean_psi')
tissue_names = cols[start_idx:end_idx]
print(f"Found {len(tissue_names)} tissues to process.")

# --- 3. PREPARE INITIAL PREDICTIONS DATAFRAME ---

# Build a single predictions DataFrame with all 56 tissue predictions
predictions = pd.DataFrame(result, columns=tissue_names)
predictions['exon_id'] = exon_ids

# --- 4. LOOP THROUGH TISSUES AND CALCULATE METRICS ---

# Initialize containers to store results from all tissues
all_psi_predictions = {'exon_id': exon_ids}
all_delta_psi_predictions = {'exon_id': exon_ids}
correlation_results = []

# Small epsilon value to prevent logit(0) or logit(1)
eps = 1e-6

print("\nProcessing tissues...")
for i, tissue in enumerate(tissue_names):
    # Merge ground truth and predictions for the current tissue
    df = pd.merge(
        ground_truth[['exon_id', 'logit_mean_psi', tissue]],
        predictions[['exon_id', tissue]],
        on='exon_id',
        suffixes=('_true', '_logit_delta_pred')
    )

    # --- PSI Splicing Calculation ---
    df['final_predicted_psi'] = expit(df[f'{tissue}_logit_delta_pred'] + df['logit_mean_psi'])
    
    # Store the final PSI prediction for this tissue
    all_psi_predictions[tissue] = df['final_predicted_psi']

    # Spearman for PSI
    valid_psi = (df[f'{tissue}_true'] >= 0) & (~df[f'{tissue}_true'].isnull())
    if valid_psi.sum() > 1: # Spearman requires at least 2 data points
        rho_psi, _ = spearmanr(df.loc[valid_psi, f'{tissue}_true']/100, df.loc[valid_psi, 'final_predicted_psi'])
    else:
        rho_psi = np.nan

    # --- Differential PSI Splicing Calculation ---
    
    # Store the predicted logit delta
    all_delta_psi_predictions[tissue] = df[f'{tissue}_logit_delta_pred']
    
    # Calculate ground-truth logit-delta for each exon
    df['tissue_frac_true'] = np.clip(df[f'{tissue}_true'] / 100, eps, 1 - eps)
    df['truth_delta_psi'] = logit(df['tissue_frac_true']) - df['logit_mean_psi']

    # Spearman for logit-delta
    valid_delta = valid_psi & (~df['logit_mean_psi'].isnull()) & (~df[f'{tissue}_logit_delta_pred'].isnull())
    if valid_delta.sum() > 1:
        rho_delta, _ = spearmanr(
            df.loc[valid_delta, 'truth_delta_psi'],
            df.loc[valid_delta, f'{tissue}_logit_delta_pred']
        )
    else:
        rho_delta = np.nan
        
    # Store results for this tissue
    correlation_results.append({
        'tissue': tissue,
        'spearman_rho_psi': rho_psi,
        'spearman_rho_delta_psi': rho_delta
    })
    
    # Print progress
    print(f"  {i+1}/{len(tissue_names)}: {tissue} | ρ(PSI)={rho_psi:.4f}, ρ(ΔPSI)={rho_delta:.4f}")


# --- 5. CONSOLIDATE AND SAVE ALL RESULTS ---

# Create final DataFrames from the collected results
psi_predictions_df = pd.DataFrame(all_psi_predictions)
delta_psi_predictions_df = pd.DataFrame(all_delta_psi_predictions)
correlations_df = pd.DataFrame(correlation_results)


# Save the DataFrames to files
psi_predictions_df.to_csv(f"{split_name}_all_tissues_predicted_psi.tsv", sep="\t", index=False)
delta_psi_predictions_df.to_csv(f"{split_name}_all_tissues_predicted_logit_delta.tsv", sep="\t", index=False)
correlations_df.to_csv(f"{split_name}_all_tissues_spearman_correlations.tsv", sep="\t", index=False)

print("\n✅ All processing complete.")
print("\n--- Summary of Spearman Correlations ---")
print(correlations_df)
print("\n--- Preview of Predicted PSI ---")
print(psi_predictions_df.head())
print("\n--- Preview of Predicted Logit Delta ---")
print(delta_psi_predictions_df.head())
print(f"\nResults saved to:\n- {split_name}_all_tissues_predicted_psi.tsv\n- {split_name}_all_tissues_predicted_logit_delta.tsv\n- {split_name}_all_tissues_spearman_correlations.tsv")

"""

########################## psi splicing ##########################

retina_pred = result[:, 0]    # First column = Retina - Eye logit(delta)

# Build predictions DataFrame
predictions = pd.DataFrame({
    'exon_id': exon_ids,
    'Retina - Eye': retina_pred
})


ground_truth = pd.read_csv(f"/home/atalukder/Contrastive_Learning/data/ASCOT/{split_name}_cassette_exons_with_logit_mean_psi.csv")

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

ground_truth = pd.read_csv(f"/home/atalukder/Contrastive_Learning/data/ASCOT/{split_name}_cassette_exons_with_logit_mean_psi.csv")

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
"""