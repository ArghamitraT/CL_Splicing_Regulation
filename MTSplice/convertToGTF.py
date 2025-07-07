import csv

input_csv = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/variable_cassette_exons.csv"
output_gtf = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/variable_cassette_exons.gtf"

with open(input_csv, newline='') as csvfile, open(output_gtf, "w") as gtffile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Parse chromosome, start, end from exon_location (e.g., "chr17:151180247–151181438")
        chrom_loc = row['exon_location'].replace('–', '-')  # fix en-dash if present
        chrom, coords = chrom_loc.split(':')
        start, end = coords.split('-')
        start = int(start) + 1  # GTF is 1-based and inclusive
        end = int(end)
        strand = row['exon_strand']
        gene_id = row['gene_id']
        exon_id = row['exon_id']
        transcript_id = exon_id  # use exon_id as transcript_id if none available
        exon_number = 1  # or infer if you have

        # Compose attributes
        attributes = f'gene_id "{gene_id}"; transcript_id "{transcript_id}"; exon_number "{exon_number}"; exon_id "{exon_id}";'
        fields = [chrom, "custom", "exon", str(start), str(end), ".", strand, ".", attributes]
        gtffile.write('\t'.join(fields) + '\n')
