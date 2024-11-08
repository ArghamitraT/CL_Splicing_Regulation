# CL_Splicing_Regulation


# Environment

```conda env create -f environment.yaml```

# Data Preprocessing

```
python scripts/preprocess_raw_data.py /data/ak5078/MSA_new alignmentknownGene.multiz100way.exonNuc_exon_intron_positions_split_file_1_IntronSeq.pkl alignmentknownGene.multiz100way.exonNuc_exon_intron_positions_split_file_2_IntronSeq.pkl alignmentknownGene.multiz100way.exonNuc_exon_intron_positions_split_file_3_IntronSeq.pkl alignmentknownGene.multiz100way.exonNuc_exon_intron_positions_split_file_4_IntronSeq.pkl
```


# Finetuning

Be sure to modify the test.sh file accordingly

```
bash scripts/run.sh
```