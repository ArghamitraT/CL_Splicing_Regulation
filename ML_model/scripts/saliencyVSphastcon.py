import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyBigWig
from scipy.stats import pearsonr, spearmanr, linregress


############################################################
#  1. Parse saliency filenames
############################################################
SAL_PATTERN = re.compile(
    r"saliency_(GT_\d+)_vs_(GT_\d+)_(GT_\d+)_(left|right)\.npy"
)

def parse_saliency_filename(fname):
    """
    Returns (exon_id, side) or None.
    """
    m = SAL_PATTERN.match(fname)
    if not m:
        return None
    anc, neg, exon_id, side = m.groups()
    return exon_id, side


############################################################
#  2. Load exon metadata from CSV
############################################################
def load_exon_metadata(df, exon_id):
    """
    Return chrom, strand, start, end for the exon.
    """
    row = df[df['exon_id'] == exon_id]
    if len(row) == 0:
        return None
    row = row.iloc[0]

    chrom  = row['chromosome']
    strand = row['exon_strand']

    start  = int(row['exon_location'].split(":")[1].split("-")[0])
    end    = int(row['exon_location'].split(":")[1].split("-")[1])

    return chrom, strand, start, end


############################################################
#  3. Reconstruct windows exactly as training did
############################################################
def calculate_5prime_intron(start, end, strand):
    if strand == '+':
        return (start - 300, start - 1)
    else:
        return (end + 1, end + 300)

def calculate_3prime_intron(start, end, strand):
    if strand == '+':
        return (end + 1, end + 300)
    else:
        return (start - 300, start - 1)


############################################################
#  4. Extract PhastCons track
############################################################
def get_phastcons(phast, chrom, start_1, end_1, strand):
    """
    Returns per-base PhastCons track aligned to model sequence orientation.
    """
    start0 = start_1 - 1
    end0   = end_1

    vals = np.array(phast.values(chrom, start0, end0))
    vals = np.nan_to_num(vals, nan=0.0)

    if strand == '-':
        vals = vals[::-1]

    return vals


############################################################
#  5. Load saliency array
############################################################
def load_saliency_vector(filepath):
    sal = np.load(filepath)
    return sal.astype(float)


############################################################
#  6. Compute correlations
############################################################
def compute_correlations(sal, phast):
    """
    Computes Spearman and Pearson.
    Also returns linear regression slope.
    """
    rho, _ = spearmanr(phast, sal)
    r, _   = pearsonr(phast, sal)
    slope, intercept, _, _, _ = linregress(phast, sal)
    return rho, r, slope, intercept


############################################################
#  7. Plot: scatter + regression line
############################################################
def plot_scatter_with_regression(
    exon_id,
    side,
    sal,
    phast,
    rho=None,
    r=None,
    slope=None,
    intercept=None,
    save_path=None
):
    # Compute regressions if not provided
    if slope is None or intercept is None:
        slope, intercept, _, _, _ = linregress(phast, sal)
    if rho is None:
        rho, _ = spearmanr(phast, sal)
    if r is None:
        r, _ = pearsonr(phast, sal)

    # Regression line
    xs = np.linspace(min(phast), max(phast), 200)
    ys = slope * xs + intercept

    # Figure
    plt.figure(figsize=(4.6, 4.4))
    plt.scatter(phast, sal, s=6, alpha=0.4, label="positions")
    plt.plot(xs, ys, color="red", lw=2)

    plt.xlabel("PhastCons", fontsize=12)
    plt.ylabel("Saliency", fontsize=12)
    plt.title(f"{exon_id} | {side}", fontsize=13)

    # Annotation box
    textstr = (
        f"Spearman ρ  = {rho:.3f}\n"
        f"Pearson r    = {r:.3f}\n"
        f"Slope m      = {slope:.3f}"
    )

    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    plt.text(
        0.05, 0.95,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox=props
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()



############################################################
#  8. Plot positional overlay (PhastCons vs saliency)
############################################################
def plot_positional_overlay(exon_id, side, sal, phast, chrom, start, end, strand, save_path=None):
    plt.figure(figsize=(14,4))
    plt.plot(sal / sal.max(), label="Normalized Saliency", alpha=0.8)
    plt.plot(phast, label="PhastCons", alpha=0.8)
    plt.title(f"{exon_id} | {side} | {chrom}:{start}-{end} ({strand})")
    plt.xlabel("Position")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


############################################################
#  9. Main driver
############################################################
def process_saliency_folder(
        sal_dir,
        csv_path,
        phast_path,
        out_dir="saliency_phastcons_plots"
    ):

    df = pd.read_csv(csv_path)
    phast = pyBigWig.open(phast_path)

    os.makedirs(out_dir, exist_ok=True)

    for fname in sorted(os.listdir(sal_dir)):
        if not fname.endswith(".npy"):
            continue

        m = re.search(r"saliency_(.+?)_(left|right)\.npy", fname)
        mid = m.group(1)

        parsed = parse_saliency_filename(fname)
        if parsed is None:
            continue

        # 4) Load saliency
        sal_path = os.path.join(sal_dir, fname)
        sal_vec = load_saliency_vector(sal_path)


        exon_id, side = parsed
        print(f"\n=== Processing {fname} ===")

        # 1) Metadata
        meta = load_exon_metadata(df, exon_id)
        if meta is None:
            print("⚠️ Exon not found:", exon_id)
            continue

        chrom, strand, start, end = meta

        # 2) Window choice
        if side == "left":
            win_start, win_end = calculate_5prime_intron(start, end, strand)
            sal_vec = sal_vec[:300]
        else:
            win_start, win_end = calculate_3prime_intron(start, end, strand)
            sal_vec = sal_vec[100:]

        # 3) Load PhastCons
        phast_vec = get_phastcons(phast, chrom, win_start, win_end, strand)

        
        if len(sal_vec) != len(phast_vec):
            print("⚠️ Length mismatch:", len(sal_vec), len(phast_vec))
            continue

        # 5) Stats
        rho, r, slope, intercept = compute_correlations(sal_vec, phast_vec)
        print(f"Spearman: {rho:.3f},  Pearson: {r:.3f},  slope: {slope:.3f}")

        # 6) Plots
        plot_scatter_with_regression(
        exon_id,
        side,
        sal_vec,
        phast_vec,
        rho=rho,
        r=r,
        slope=slope,
        intercept=intercept,
        save_path=f"{out_dir}/{mid}_{side}_scatter.png"
    )


        plot_positional_overlay(
            exon_id,
            side,
            sal_vec,
            phast_vec,
            chrom,
            win_start,
            win_end,
            strand,
            save_path=f"{out_dir}/{mid}_{side}_overlay.png"
        )


############################################################
#  ENTRY POINT
############################################################

if __name__ == "__main__":
    main_dir = "/gpfs/commons/home/atalukder/Contrastive_Learning"
    trimester = time.strftime("_%Y_%m_%d__%H_%M_%S")
    process_saliency_folder(
        sal_dir= f"{main_dir}/data/extra/arrays/contrast_saliency_random/arrays/contrast_saliency_random/",
        csv_path=f"{main_dir}/data/final_data/ASCOT_finetuning/variable_cassette_exons_with_logit_mean_psi.csv",
        phast_path=f"{main_dir}/data/multiz100way/phastcon_score/existing_score/hg38.phastCons100way.bw",
        out_dir=f"{main_dir}/data/extra/phastcons_analysis_plots/F_{trimester}"
    )
