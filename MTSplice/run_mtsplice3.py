from mmsplice.mtsplice import MTSplice
from mmsplice.utils import encodeDNA
import pandas as pd
import pickle


from mmsplice.vcf_dataloader import SplicingVCFDataloader
from mmsplice import MMSplice, predict_save, predict_all_table
# from mmsplice.utils import max_varEff

main_dir = "/home/atalukder/Contrastive_Learning/models/MMSplice_MTSplice-master/"
# example files
# gtf = main_dir+'tests/data/test.gtf'
# vcf = main_dir+'tests/data/test.vcf.gz'
# fasta = main_dir+'tests/data/hg19.nochr.chr17.fa'
# csv = main_dir+'pred.csv'

gtf = '/home/atalukder/Contrastive_Learning/data/MTSplice/variable_cassette_exons.gtf'
vcf = '/home/atalukder/Contrastive_Learning/data/MTSplice/dummy_new.vcf.gz'
fasta = '/home/atalukder/Contrastive_Learning/data/MTSplice/hg38.fa'

model = MMSplice()
dl = SplicingVCFDataloader(gtf, fasta, vcf, tissue_specific=True)

predictions = predict_all_table(model, dl, pathogenicity=True, splicing_efficiency=True)

print()

# from scipy.special import logit
# from scipy.stats import spearmanr
# import numpy as np

# # Step 1: Extract matching ground-truth psi values
# retina_df["psi_val"] = retina_df.index.map(lambda k: data[k]["psi_val"])

# # Step 2: Convert both to numpy arrays
# y_pred_logit = retina_df["Retina_logit_psi"].values
# y_true_raw = retina_df["psi_val"].values  # Assumes in percent [0–100]

# # Step 3: Convert PSI to logit(ψ), with clipping to avoid logit(0) or logit(1)
# eps = 1e-6
# y_true_logit = logit(np.clip(y_true_raw / 100, eps, 1 - eps))

# # Step 4: Compute Spearman correlation
# rho, _ = spearmanr(y_true_logit, y_pred_logit)

# print(f"\n🔬 Spearman ρ (Retina logit PSI): {rho:.4f}")


# retina_df.to_csv("mtsplice_retina_predictions.tsv", sep="\t")



