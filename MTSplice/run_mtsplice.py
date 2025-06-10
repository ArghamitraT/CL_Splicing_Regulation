# === Imports ===
# from mmsplice.mtsplice import MTSplice  # TSplice model in code
# # # from mmsplice.utils import encodeDNA
# # # from mmsplice.exon_dataloader import read_fasta
# # from scipy.special import expit  # sigmoid for natural scale
# # import pandas as pd

# # # === Sequence preparation ===
# # main_dir = "/home/atalukder/Contrastive_Learning/models/MMSplice_MTSplice-master/"
# # fasta = main_dir + 'tests/data/hg19.nochr.chr17.fa'
# # gtf = main_dir + 'tests/data/test.gtf'

# # Get an example exon sequence to test
# # Use the `read_fasta` helper and your own exon-coordinates to extract the sequence.
# # For now, hardcoding an example exon sequence:
# # example_seq = "A" * 300 + "C" * 100 + "G" * 100 + "T" * 300  # replace with real exon+flank

# # # === MTSplice model ===
# # mtsplice = MTSplice(deep=True)  # or deep=False

# # # # Predict tissue-specific logit(Ψ)
# # logit_psi = mtsplice.predict(example_seq)  # shape: (1, 56)

# # # Convert to PSI (natural scale)
# # psi = expit(logit_psi)

# # # Make a readable DataFrame
# # from mmsplice.mtsplice import TISSUES
# # psi_df = pd.DataFrame(psi, columns=TISSUES)
# # print(psi_df.T.sort_values(by=0, ascending=False))  # top tissues with highest inclusion


# # exons = [
# #     "A" * 300 + "C" * 100 + "G" * 100 + "T" * 300,
# #     "T" * 300 + "G" * 100 + "C" * 100 + "A" * 300,
# # ]

# # mtsplice = MTSplice(deep=True)

# # # Call model.predict() on each exon
# # psi = [expit(mtsplice.predict(seq)) for seq in exons]

# # # Combine into a DataFrame
# # import pandas as pd
# # from mmsplice.mtsplice import TISSUES
# # df = pd.DataFrame(psi, columns=TISSUES, index=[f"exon_{i}" for i in range(len(exons))])
# # print(df.T)  # View PSI per tissue per exon


from mmsplice.vcf_dataloader import SplicingVCFDataloader
from mmsplice.mtsplice import MTSplice
from scipy.special import expit

main_dir = "/home/atalukder/Contrastive_Learning/models/MMSplice_MTSplice-master/"
# example files
gtf = main_dir+'tests/data/test.gtf'
vcf = main_dir+'tests/data/test.vcf.gz'
fasta = main_dir+'tests/data/hg19.nochr.chr17.fa'
# csv = main_dir+'pred.csv'


# # # Files you need
# # gtf = "your.gtf"
# # fasta = "your.fa"
# # vcf = "dummy.vcf"  # Must exist, but you won't use the alt sequences



# # Initialize MTSplice
mtsplice = MTSplice(deep=True)

# # Load exon sequences using the dataloader
dl = SplicingVCFDataloader(gtf, fasta, vcf, encode=False, tissue_specific=True, tissue_overhang=(200, 200))

# Predict Ψ for each exon (reference only)
for batch in dl:
    ref_seq = batch['inputs']  # only use reference
    logit_psi = mtsplice.predict(ref_seq)
    psi = expit(logit_psi)
    print(psi)  # Ψ across 56 tissues
    break  # remove to loop over all exons





#####################################
# # import tensorflow as tf
# # print("GPUs available:", tf.config.list_physical_devices('GPU'))


# # Import

# from help import print_stuff

# print_stuff("hello")

from mmsplice.vcf_dataloader import SplicingVCFDataloader
from mmsplice import MMSplice, predict_save, predict_all_table
# from mmsplice.utils import max_varEff

main_dir = "/home/atalukder/Contrastive_Learning/models/MMSplice_MTSplice-master/"
# example files
gtf = main_dir+'tests/data/test.gtf'
vcf = main_dir+'tests/data/test.vcf.gz'
fasta = main_dir+'tests/data/hg19.nochr.chr17.fa'
csv = main_dir+'pred.csv'

# # dl = SplicingVCFDataloader(gtf, fasta, vcf, encode=False, split_seq=True)

model = MMSplice()
dl = SplicingVCFDataloader(gtf, fasta, vcf)

predictions = predict_all_table(model, dl, pathogenicity=True, splicing_efficiency=True)

# predictionsMax = max_varEff(predictions)

# predictionsMax.sort_values(['delta_logit_psi']).head()



# # next(dl)




# # dl = SplicingVCFDataloader(gtf, fasta, vcf, tissue_specific=False)
# # dl = SplicingVCFDataloader(gtf, fasta, vcf, tissue_specific=True)

# # # Specify model
# # model = MMSplice()

# # # Or predict and return as df
# # predictions = predict_all_table(model, dl, pathogenicity=True, splicing_efficiency=True)

print("all done")