import os
import re
import numpy as np
import matplotlib.pyplot as plt
import time

timestamp = time.strftime("_%Y_%m_%d__%H_%M_%S")

# ---------------------------------------------------------
# NEW: 3b. Load and CONCAT saliency left+right for full window
# ---------------------------------------------------------

def load_and_concat_saliency_vectors(sal_dir):
    HIGH_full, LOW_full = [], []

    for fname in os.listdir(sal_dir):
        if not fname.endswith(".npy"):
            continue

        parsed = parse_contrastive_saliency_filename(fname)
        if parsed is None:
            continue

        A, N, TARGET, side = parsed
        cls = classify_saliency(A, N, TARGET)

        sal = np.load(os.path.join(sal_dir, fname))

        # Temporarily store separately — we need to merge left+right later
        key = (A, N, TARGET)

        if "_left" in fname:
            key_side = "left"
        else:
            key_side = "right"

        # Store intermediate
        yield key, key_side, cls, sal


def merge_left_and_right(left_dict, right_dict):
    """
    Merge left and right saliency for each (A,N,TARGET).
    """
    merged = []
    for key in left_dict:
        if key not in right_dict:
            continue

        left = left_dict[key]
        right = right_dict[key]

        full = np.concatenate([left, right])
        merged.append(full)

    return merged


# ---------------------------------------------------------
# NEW: 4b. Compute mean FULL-sequence saliency profiles
# ---------------------------------------------------------

def compute_mean_full_profiles(HIGH_full, LOW_full):
    HIGH_full_mean = np.mean(np.vstack(HIGH_full), axis=0) if HIGH_full else None
    LOW_full_mean  = np.mean(np.vstack(LOW_full), axis=0)  if LOW_full  else None
    return HIGH_full_mean, LOW_full_mean


# ---------------------------------------------------------
# NEW: 5b. Plot FULL combined profiles
# ---------------------------------------------------------

def plot_mean_full_profile(HIGH_full_mean, LOW_full_mean, save_path=None):
    plt.figure(figsize=(12, 4))
    plt.plot(HIGH_full_mean, label="HIGH (functional anchors)", color="dodgerblue")
    plt.plot(LOW_full_mean,  label="LOW (negatives)",          color="darkorange")

    # Mark splice junctions visually
    # junction1 = left length (300)
    # junction2 = left + exon_start + exon_end + right, but exon portion is included already

    plt.axvline(300, color="black", linestyle="--", alpha=0.5, label="5′ splice site")
    plt.axvline(len(HIGH_full_mean)-300, color="black", linestyle="--", alpha=0.5, label="3′ splice site")

    plt.title("Mean FULL saliency profile (5′ intron → exon → 3′ intron)")
    plt.xlabel("Position along concatenated sequence")
    plt.ylabel("Saliency")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


# ---------------------------------------------------------
# 1. Parse filename: A vs N and TARGET assignment
# ---------------------------------------------------------

def parse_contrastive_saliency_filename(fname):
    """
    Example:
        saliency_GT_03764_vs_GT_23158_GT_03764_left.npy
    Returns:
        A = GT_03764
        N = GT_23158
        TARGET = GT_03764
        side = left
    """
    pat = r"saliency_(GT_\d+)_vs_(GT_\d+)_(GT_\d+)_(left|right)\.npy"
    m = re.match(pat, fname)
    if not m:
        return None
    A, N, TARGET, side = m.groups()
    return A, N, TARGET, side


# ---------------------------------------------------------
# 2. Decide whether the saliency belongs to A (HIGH) or N (LOW)
# ---------------------------------------------------------

def classify_saliency(A, N, TARGET):
    return "HIGH" if TARGET == A else "LOW"


# ---------------------------------------------------------
# 3. Load saliency vectors from directory and group them
# ---------------------------------------------------------

def load_saliency_vectors_by_class(sal_dir):
    HIGH_left, LOW_left = [], []
    HIGH_right, LOW_right = [], []

    for fname in os.listdir(sal_dir):
        if not fname.endswith(".npy"):
            continue
        
        parsed = parse_contrastive_saliency_filename(fname)
        if parsed is None:
            continue
        
        A, N, TARGET, side = parsed
        cls = classify_saliency(A, N, TARGET)

        sal = np.load(os.path.join(sal_dir, fname))

        # Extract intron portion only (first 300 or last 300)
        if mode == "intronOnly":
            if side == "left":
                if overhang == 200:
                    intron = sal[-200:]
                else:
                    intron = sal
            
                if cls == "HIGH":
                    HIGH_left.append(intron)
                else:
                    LOW_left.append(intron)

            else:  # right
                
                if overhang == 200:
                    intron = sal[:200]
                else:
                    intron = sal
        
                if cls == "HIGH":
                    HIGH_right.append(intron)
                else:
                    LOW_right.append(intron)
        else:
            if side == "left":
                if overhang == 200:
                    intron = sal[-300:]
                else:
                    intron = sal
            
                if cls == "HIGH":
                    HIGH_left.append(intron)
                else:
                    LOW_left.append(intron)

            else:  # right
                
                if overhang == 200:
                    intron = sal[:300]
                else:
                    intron = sal
        
                if cls == "HIGH":
                    HIGH_right.append(intron)
                else:
                    LOW_right.append(intron)

        


    return HIGH_left, LOW_left, HIGH_right, LOW_right


# ---------------------------------------------------------
# 4. Compute mean saliency profiles
# ---------------------------------------------------------

def compute_mean_profiles(HIGH_left, LOW_left, HIGH_right, LOW_right):
    HIGH_left_mean  = np.mean(np.vstack(HIGH_left), axis=0)  if HIGH_left  else None
    LOW_left_mean   = np.mean(np.vstack(LOW_left), axis=0)   if LOW_left   else None
    HIGH_right_mean = np.mean(np.vstack(HIGH_right), axis=0) if HIGH_right else None
    LOW_right_mean  = np.mean(np.vstack(LOW_right), axis=0)  if LOW_right  else None

    return HIGH_left_mean, LOW_left_mean, HIGH_right_mean, LOW_right_mean


# ---------------------------------------------------------
# 5. Plot profiles
# ---------------------------------------------------------

def plot_mean_profiles(HIGH_left_mean, LOW_left_mean, HIGH_right_mean, LOW_right_mean):
    # ------ Left (5′ intron) ------
    plt.figure(figsize=(8, 4))
    plt.plot(HIGH_left_mean, label="HIGH (functional anchors)", color="dodgerblue")
    plt.plot(LOW_left_mean, label="LOW (negatives)", color="darkorange")

    plt.axvline(300, color='k', linestyle='--', alpha=0.5)
    plt.title("Mean saliency profile (5′ intron)")
    plt.xlabel("Position relative to exon start (intron → exon)")
    plt.ylabel("Saliency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{sal_dir}/figures/saliency_profile_5prime_{timestamp}.png")

    # ------ Right (3′ intron) ------
    plt.figure(figsize=(8, 4))
    plt.plot(HIGH_right_mean, label="HIGH (functional anchors)", color="dodgerblue")
    plt.plot(LOW_right_mean, label="LOW (negatives)", color="darkorange")

    plt.axvline(0, color='k', linestyle='--', alpha=0.5)
    plt.title("Mean saliency profile (3′ intron)")
    plt.xlabel("Position relative to exon end (exon → intron)")
    plt.ylabel("Saliency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{sal_dir}/figures/saliency_profile_3prime_{timestamp}.png")


# ---------------------------------------------------------
# 6. Full pipeline wrapper
# ---------------------------------------------------------

def run_saliency_anchor_negative_analysis(sal_dir):
    sal_dirWarrays = f"{sal_dir}/arrays"
    HIGH_left, LOW_left, HIGH_right, LOW_right = load_saliency_vectors_by_class(sal_dirWarrays)
    profiles = compute_mean_profiles(HIGH_left, LOW_left, HIGH_right, LOW_right)
    plot_mean_profiles(*profiles)


def run_full_sequence_saliency_analysis(sal_dir):
    sal_arrays = f"{sal_dir}/arrays"

    left_dict = {}
    right_dict = {}
    class_map = {}

    # First pass: Load everything
    for key, side, cls, sal in load_and_concat_saliency_vectors(sal_arrays):
        class_map[key] = cls

        if side == "left":
            left_dict[key] = sal
        else:
            right_dict[key] = sal

    # Merge left+right for each exon
    HIGH_full, LOW_full = [], []

    merged = merge_left_and_right(left_dict, right_dict)

    for key, full_sal in zip(left_dict.keys(), merged):
        cls = class_map[key]
        if cls == "HIGH":
            HIGH_full.append(full_sal)
        else:
            LOW_full.append(full_sal)

    # Compute mean profiles
    HIGH_full_mean, LOW_full_mean = compute_mean_full_profiles(HIGH_full, LOW_full)

    import pickle

    # Save directory for pickles
    pkl_dir = f"{sal_dir}/pickles"
    os.makedirs(pkl_dir, exist_ok=True)

    save_data = {
        "HIGH_full_mean": HIGH_full_mean,
        "LOW_full_mean": LOW_full_mean,
        "HIGH_count": len(HIGH_full),
        "LOW_count": len(LOW_full),
        "timestamp": timestamp,
    }

    pkl_path = f"{pkl_dir}/full_saliency_means_{timestamp}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(save_data, f)

    print(f"[💾] Saved mean full saliency profiles → {pkl_path}")



    # Plot
    out_path = f"{sal_dir}/figures/full_saliency_profile_{timestamp}.png"
    plot_mean_full_profile(HIGH_full_mean, LOW_full_mean, save_path=out_path)


def run_saliency_analysis_all(sal_dir):
    # Run original separate left+right analysis
    # run_saliency_anchor_negative_analysis(sal_dir)

    # Run new full-sequence analysis
    run_full_sequence_saliency_analysis(sal_dir)


# ---------------------------------------------------------
# 7. Run the analysis
# ---------------------------------------------------------
main_dir = "/gpfs/commons/home/atalukder/Contrastive_Learning"
sal_dir = f"{main_dir}/data/extra/contrast_saliency_random/f__2025_11_19__13_37_51/"

overhang = 200  # Set to 200 for 200bp data, or 300 for 300bp data
mode = "exon"  # Set to "intronOnly" for intron-only data, or "full" for full-sequence data
run_saliency_analysis_all(sal_dir)

