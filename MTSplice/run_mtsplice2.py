from mmsplice.mtsplice import MTSplice
from mmsplice.utils import encodeDNA
import pandas as pd
import pickle


from mmsplice.vcf_dataloader import SplicingVCFDataloader
from mmsplice import MMSplice, predict_save, predict_all_table

main_dir = "/home/atalukder/Contrastive_Learning/models/MMSplice_MTSplice-master/"

gtf = '/home/atalukder/Contrastive_Learning/data/MTSplice/variable_cassette_exons.gtf'
vcf = '/home/atalukder/Contrastive_Learning/data/MTSplice/variable_cassette_exons.vcf.gz'
fasta = '/home/atalukder/Contrastive_Learning/data/MTSplice/hg38.fa'

model = MMSplice()
dl = SplicingVCFDataloader(gtf, fasta, vcf, tissue_specific=True)

predictions = predict_all_table(model, dl, pathogenicity=True, splicing_efficiency=True)

print()

import pandas as pd
import numpy as np
from scipy.special import logit, expit
from scipy.stats import spearmanr

########################## psi splicing ##########################

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

# ---- Save results ----
# df[['exon_id', f'{tissue}_true', 'logit_mean_psi', 'logit_delta_pred', 'final_predicted_psi']].to_csv(
#     f"tsplice_final_predictions_{tissue.replace(' ', '_')}.tsv", sep="\t", index=False
# )
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