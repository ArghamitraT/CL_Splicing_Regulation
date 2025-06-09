

# import tensorflow as tf
# print("GPUs available:", tf.config.list_physical_devices('GPU'))


# Import
from mmsplice.vcf_dataloader import SplicingVCFDataloader
from mmsplice import MMSplice, predict_save, predict_all_table
from mmsplice.utils import max_varEff

main_dir = "/home/atalukder/Contrastive_Learning/models/MMSplice_MTSplice-master/"
# example files
gtf = main_dir+'tests/data/test.gtf'
vcf = main_dir+'tests/data/test.vcf.gz'
fasta = main_dir+'tests/data/hg19.nochr.chr17.fa'
csv = main_dir+'pred.csv'

dl = SplicingVCFDataloader(gtf, fasta, vcf, encode=False, split_seq=True)

model = MMSplice()
dl = SplicingVCFDataloader(gtf, fasta, vcf)

predictions = predict_all_table(model, dl, pathogenicity=True, splicing_efficiency=True)

predictionsMax = max_varEff(predictions)

predictionsMax.sort_values(['delta_logit_psi']).head()



# next(dl)




# dl = SplicingVCFDataloader(gtf, fasta, vcf, tissue_specific=False)
# dl = SplicingVCFDataloader(gtf, fasta, vcf, tissue_specific=True)

# # Specify model
# model = MMSplice()

# # Or predict and return as df
# predictions = predict_all_table(model, dl, pathogenicity=True, splicing_efficiency=True)

print("all done")