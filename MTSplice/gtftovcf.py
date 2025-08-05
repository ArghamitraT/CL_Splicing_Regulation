import csv

gtf_file = "/home/atalukder/Contrastive_Learning/data/MTSplice/variable_cassette_exons.gtf"
vcf_out = "/home/atalukder/Contrastive_Learning/data/MTSplice/variable_cassette_exons.vcf"


header_file = "/home/atalukder/Contrastive_Learning/data/MTSplice/vcf_header.txt"


# The template INFO string you want for every record
template_info = (
    "ALLELEID=247288;"
    "CLNDISDB=MedGen:C0027672,SNOMED_CT:699346009|MedGen:C0677776,Orphanet:ORPHA145|MedGen:C2676676,OMIM:604370|MedGen:CN169374;"
    "CLNDN=Hereditary_cancer-predisposing_syndrome|Hereditary_breast_and_ovarian_cancer_syndrome|Breast-ovarian_cancer,_familial_1|not_specified;"
    "CLNHGVS=NC_000017.10:g.41201138T>G;"
    "CLNREVSTAT=reviewed_by_expert_panel;"
    "CLNSIG=Likely_benign;"
    "CLNVC=single_nucleotide_variant;"
    "CLNVCSO=SO:0001483;"
    "GENEINFO=BRCA1:672;"
    "MC=SO:0001627|intron_variant,SO:0001819|synonymous_variant;"
    "ORIGIN=1;"
    "RS=879255493"
)

with open(header_file, "r") as header, open(vcf_out, "w") as out:
    for line in header:
        out.write(line if line.endswith("\n") else line + "\n")

    with open(gtf_file, newline='') as gtf:
        reader = csv.reader(gtf, delimiter='\t')
        for row in reader:
            if len(row) < 9: continue
            chrom, source, feature, start, end, score, strand, frame, attributes = row
            if feature != "exon": continue

            pos = int(start)
            ref, alt = "T", "G"  # can match your template or set arbitrarily
            id_field = "252887"  # you can set to "." if you want
            qual = "."
            filt = "."

            # Write the line with your desired INFO
            out.write(f"{chrom}\t{pos}\t{id_field}\t{ref}\t{alt}\t{qual}\t{filt}\t{template_info}\n")