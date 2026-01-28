import os
import sys
import time
import random
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.linear_model import LogisticRegression
import joblib
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
import hydra


def find_contrastive_root(start: Path = Path(__file__)) -> Path:
    for parent in start.resolve().parents:
        if parent.name == "Contrastive_Learning":
            return parent
    raise RuntimeError("Could not find 'Contrastive_Learning' directory.")

# Set env var *before* hydra loads config
os.environ["CONTRASTIVE_ROOT"] = str(find_contrastive_root())
CONTRASTIVE_ROOT = find_contrastive_root()


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.config import print_config
from src.datasets.utility import get_windows_with_padding


trimester = time.strftime("_%Y_%m_%d__%H_%M_%S")

# os.environ['WANDB_INIT_TIMEOUT'] = '600'
def get_optimal_num_workers():
    num_cpus = os.cpu_count()
    num_gpus = torch.cuda.device_count()
    return min(num_cpus // max(1, num_gpus), 8)


# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import print_config
from src.model.lit import create_lit_model
from src.datasets.lit import ContrastiveIntronsDataModule


######### parameters #############
result_dir = "exprmnt_2025_10_26__14_29_04"
######### parameters ##############
# exprmnt_2025_05_04__11_29_05

def get_best_checkpoint(config):
    # simclr_ckpt = f"{root_path}/files/results/{result_dir}/weights/checkpoints/introns_cl/{config.embedder._name_}/199/best-checkpoint.ckpt"
    return f"{str(CONTRASTIVE_ROOT)}/files/results/{result_dir}/weights/checkpoints/introns_cl/{config.embedder._name_}/{config.dataset.seq_len}/best-checkpoint.ckpt"
    # return str(CONTRASTIVE_ROOT / "files/results/exprmnt_2025_05_04__11_29_05/weights/checkpoints/introns_cl/ResNet1D/199/best-checkpoint.ckpt")


def get_config_path():
    # simclr_ckpt = f"{root_path}/files/results/{result_dir}/weights/checkpoints/introns_cl/{config.embedder._name_}/199/best-checkpoint.ckpt"
    return f"{str(CONTRASTIVE_ROOT)}/files/results/{result_dir}/files/configs/"
    # return str(CONTRASTIVE_ROOT / "files/results/exprmnt_2025_05_04__11_29_05/weights/checkpoints/introns_cl/ResNet1D/199/best-checkpoint.ckpt")



def load_pretrained_model(config, device):
    model = create_lit_model(config)
    ckpt = torch.load(get_best_checkpoint(config), map_location=device)
    state_dict = ckpt["state_dict"]
    cleaned_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model.model.encoder


# ---------------- logging ----------------
import os, sys, time, json, logging, numpy as np
def setup_logging(save_dir: Path, filename: str = "contrastive_saliency.log"):
    save_dir.mkdir(parents=True, exist_ok=True)
    p = save_dir / filename
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    fh = logging.FileHandler(p, mode="w"); fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt); ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh); logger.addHandler(ch)
    logging.info(f"Logging to {p}")



def plot_saliency_separate_windows(
    sal_left: np.ndarray,
    sal_right: np.ndarray,
    seq_entry: dict,
    exon_id: str,
    out_prefix: str,
    tissue_acceptor_intron: int = 300,
    tissue_donor_exon: int = 100,
    tissue_donor_intron: int = 300,
):
    """
    Plot two separate saliency barplots for seql (5′) and seqr (3′) windows.

    - seql: 5′ intron + exon_start
    - seqr: exon_end + 3′ intron
    The true exon portion is highlighted in pink; the rest of the window’s exon context is red.

    Args:
        sal_left, sal_right: saliency arrays for acceptor and donor sides
        seq_entry: dict with keys "5p", "3p", and "exon" -> {"start", "end"}
        exon_id: exon identifier string
        out_prefix: base path for saving (two figures will be saved)
    """

    # --- 1️⃣ Extract sequences and lengths ---
    seq_5p = seq_entry["5p"]
    seq_3p = seq_entry["3p"]
    exon_mid = int(len(seq_entry["exon"])/2)
    # exon_start_seq = seq_entry["exon"].get("start", "")
    # exon_end_seq = seq_entry["exon"].get("end", "")
    exon_start_seq = seq_entry["exon"][:exon_mid]
    exon_end_seq = seq_entry["exon"][exon_mid:]


    len_5p_real = len(seq_5p)
    len_3p_real = len(seq_3p)
    len_exon_start_real = len(exon_start_seq)
    len_exon_end_real = len(exon_end_seq)

    # --- 2️⃣ Decide offset for shorter exons ---
    if len_exon_start_real < 100 or len_exon_end_real < 100:
        offset = 100
    else:
        offset = len_exon_start_real

    # --- 3️⃣ Biological slicing ---
    sal_left_trimmed = sal_left[
        - tissue_acceptor_intron +len_5p_real :
        tissue_acceptor_intron + offset
    ]
    sal_right_trimmed = sal_right[
        tissue_donor_exon - offset :
        tissue_donor_exon + len_3p_real
    ]

    # --- 4️⃣ Normalize ---
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    sal_left_trimmed = normalize(sal_left_trimmed)
    sal_right_trimmed = normalize(sal_right_trimmed)

    # --- 5️⃣ Define exon boundary indices ---
    # For seql (5′)
    exon_start_idx_L =  len_5p_real
    exon_end_idx_L = exon_start_idx_L + len_exon_start_real
    exon_start_idx_R = tissue_donor_exon-(len_exon_end_real)
    exon_end_idx_R = tissue_donor_exon



    # --- Combined subplot for 5′ and 3′ windows ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 3), sharey=True)

    # -----------------------
    # 5′ window (seql)
    # -----------------------
    positions_L = np.arange(len(sal_left_trimmed))
    axes[0].bar(positions_L, sal_left_trimmed, color="dimgray", width=1.0, linewidth=0)
    axes[0].axvspan(0, exon_start_idx_L, color="lightblue", alpha=0.3, label="5′ Intron")
    axes[0].axvspan(exon_start_idx_L, exon_end_idx_L, color="pink", alpha=0.5, label="Exon (true)")
    axes[0].axvspan(exon_end_idx_L, len(sal_left_trimmed), color="red", alpha=0.3, label="Exon (context)")
    axes[0].set_title(f"5′ Window (seql) — {exon_id}", fontsize=13)
    axes[0].set_xlabel("Position (5′→Exon start)")
    axes[0].set_ylabel("Normalized saliency")
    axes[0].legend(frameon=False, loc="upper right")

    # -----------------------
    # 3′ window (seqr, reversed axis)
    # -----------------------
    positions_R = np.arange(len(sal_right_trimmed))
    axes[1].bar(positions_R, sal_right_trimmed, color="dimgray", width=1.0, linewidth=0)
    axes[1].axvspan(0, exon_start_idx_R, color="red", alpha=0.3, label="Exon (context)")
    axes[1].axvspan(exon_start_idx_R, exon_end_idx_R, color="pink", alpha=0.5, label="Exon (true)")
    axes[1].axvspan(exon_end_idx_R, len(sal_right_trimmed), color="lightgreen", alpha=0.3, label="3′ Intron")
    axes[1].invert_xaxis()
    axes[1].set_title(f"3′ Window (seqr, reversed) — {exon_id}", fontsize=13)
    axes[1].set_xlabel("Position (3′ intron → Exon end)")
    axes[1].legend(frameon=False, loc="upper right")

    # -----------------------
    # Save and close
    # -----------------------
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_both_{trimester}.png", dpi=200)
    plt.close()

    print(f"[✅] Saved seql and seqr saliency plots for {exon_id}")


def get_exon_embedding(data_module, exon_id: str, loader_type: str):
    """
    Retrieve a batch containing a given exon_id from PSIRegressionDataModule.
    Returns the batch tuple (view0, view1, psi_vals, exon_ids), and the index
    of the exon_id within that batch.
    """

    if loader_type == "train":
        loader = data_module.train_dataloader()
    elif loader_type == "val":
        loader = data_module.val_dataloader()
    else:
        loader = data_module.test_dataloader()
    
    dataset = loader.dataset
    tokenizer = data_module.tokenizer
    exon_list = [exon_id]
    os.makedirs(main_dir, exist_ok=True)

    for exon_name in exon_list:
        if exon_name not in dataset.data:
            print(f"⚠️ Skipping {exon_name}: not found in dataset.")
            continue

        # expr_group = "High" if exon_name in high_expr_exons else "Low"

        all_views_dict = dataset.data[exon_name]
        species_names = list(all_views_dict.keys())

        species_names = ['hg38', 'panTro4']
        entry = []
        seqs_arr = []
        for sp in species_names:
            seqs = all_views_dict[sp]
            if not all(k in seqs for k in ["5p", "exon", "3p"]):
                print(f"⚠️ Skipping {sp} for {exon_name}: missing 5p/3p/exon keys")
                continue

            full_seq = seqs["5p"] + seqs["exon"] + seqs["3p"]
            windows = get_windows_with_padding(
                dataset.tissue_acceptor_intron,
                dataset.tissue_donor_intron,
                dataset.tissue_acceptor_exon,
                dataset.tissue_donor_exon,
                full_seq,
                overhang=(dataset.len_3p, dataset.len_5p)
            )

            seql = tokenizer([windows["acceptor"]]).float()
            seqr = tokenizer([windows["donor"]]).float()
            entry.append([seql, seqr])
            seqs_arr.append(seqs)
            # with torch.no_grad():
            #     if sp == "hg38":
            #         zA = model(seql.to(device), seqr.to(device)).cpu().numpy()
            #     else:
            #         zP = model(seql.to(device), seqr.to(device)).cpu().numpy()
            
    return entry, seqs_arr


def save_saliency_and_entry(base_path: str, exon_id: str, sal_left, sal_right, seq_entry):
    """
    Save saliency arrays and their corresponding sequence entry as a single .npz file.
    
    Args:
        base_path (str): Base path for saving (e.g., '/path/to/arrays/')
        exon_id (str): Exon identifier used in filename.
        sal_left (np.ndarray): Left-side saliency array.
        sal_right (np.ndarray): Right-side saliency array.
        seq_entry (dict): Entry dictionary containing 5p, 3p, exon info.
    """
    os.makedirs(base_path, exist_ok=True)

    save_path = os.path.join(base_path, f"{exon_id}_saliency_entry.npz")
    np.savez_compressed(
        save_path,
        sal_left=sal_left,
        sal_right=sal_right,
        seq_entry=np.array(seq_entry, dtype=object)  # store as generic object array
    )
    print(f"[💾] Saved saliency + entry for {exon_id} → {save_path}")


def aggregate_saliency(x: torch.Tensor, g: torch.Tensor, multiply_by_input: bool = True) -> np.ndarray:
    # Supports (B, L, C) or (B, C, L)
    if x.shape != g.shape:
        raise RuntimeError(f"Shape mismatch: input {x.shape} vs grad {g.shape}")
    if x.dim() != 3:
        raise RuntimeError("Expect 3D input (B,L,C) or (B,C,L).")
    # Option A: channels-last
    sal1 = ((x * g) if multiply_by_input else g).abs().sum(dim=-1)  # (B,L)
    # Option B: channels-first
    sal2 = ((x * g) if multiply_by_input else g).abs().sum(dim=1)   # (B,L)
    # Choose the one whose L matches the larger spatial dimension
    L_guess = max(x.shape[1], x.shape[2])
    sal = sal1 if sal1.shape[1] == L_guess else sal2
    return sal.mean(dim=0).detach().cpu().numpy()

def normalize_01(a: np.ndarray) -> np.ndarray:
    if a.size == 0: return a
    lo, hi = float(a.min()), float(a.max())
    return np.zeros_like(a) if hi == lo else (a - lo) / (hi - lo)


def compute_contrastive_margin(zA, zP, zN, sim_type="cosine"):
    """Compute s(a,p) - s(a,n)."""
    if sim_type == "cosine":
        s_ap = F.cosine_similarity(zA, zP, dim=-1).mean()
        s_an = F.cosine_similarity(zA, zN, dim=-1).mean()
    else:
        s_ap = (zA * zP).sum(dim=-1).mean()
        s_an = (zA * zN).sum(dim=-1).mean()
    return s_ap - s_an, s_ap, s_an


def compute_and_save_saliency_pair(anchor_id, negative_id, encoder, data_module, device, main_dir, timestamp):
    """Compute saliency between two exons, skipping gracefully if one is missing."""
    
    combo_name = f"{anchor_id}_vs_{negative_id}"

    # --- 1️⃣ Try fetching both batches safely ---
    try:
        entry_A, seq_entry_A = get_exon_embedding(data_module, anchor_id, loader_type="train")
    except Exception as e:
        logging.warning(f"⚠️ Skipping pair {combo_name}: could not fetch anchor exon {anchor_id} — {e}")
        return

    try:
        entry_N, seq_entry_N = get_exon_embedding(data_module, negative_id, loader_type="train")
    except Exception as e:
        logging.warning(f"⚠️ Skipping pair {combo_name}: could not fetch negative exon {negative_id} — {e}")
        return
    
    try:
        xA_left  = entry_A[0][0].detach().clone().requires_grad_(True)
        xA_right = entry_A[0][1].detach().clone().requires_grad_(True)
        
        xP_left  = entry_A[1][0].detach().clone().requires_grad_(True)
        xP_right = entry_A[1][1].detach().clone().requires_grad_(True)

        xN_left  = entry_N[0][0].detach().clone().requires_grad_(True)
        xN_right = entry_N[0][1].detach().clone().requires_grad_(True)
    except Exception as e:
        logging.warning(f"⚠️ Could not extract raw backbone inputs for {combo_name}: {e}")
        return

    try:
        zA = encoder(xA_left.to(device), xA_right.to(device))
        zP = encoder(xP_left.to(device), xP_right.to(device))   # positive = anchor (placeholder)
        zN = encoder(xN_left.to(device), xN_right.to(device))
    except Exception as e:
        logging.warning(f"⚠️ Encoder forward failed for {combo_name}: {e}")
        return

    
    # try:
    #     margin, s_ap, s_an = compute_contrastive_margin(zA, zP, zN)
    #     encoder.zero_grad(set_to_none=True)
    #     # clear existing grads
    #     for t in (xA_left, xA_right, xN_left, xN_right):
    #         if t.grad is not None:
    #             t.grad.zero_()
    #     margin.backward()

    # except Exception as e:
    #     logging.warning(f"⚠️ Encoder forward/backward failed for {combo_name}: {e}")
    #     return


    try:
        margin, s_ap, s_an = compute_contrastive_margin(zA, zP, zN)
        encoder.zero_grad(set_to_none=True)
        # clear existing grads
        for t in (xA_left, xA_right, xN_left, xN_right):
            if t.grad is not None:
                t.grad.zero_()
        s_an.backward()

    except Exception as e:
        logging.warning(f"⚠️ Encoder forward/backward failed for {combo_name}: {e}")
        return

    
    # --- 4️⃣ Aggregate saliency ---
    try:
        sal_A_left  = normalize_01(aggregate_saliency(xA_left,  xA_left.grad,  multiply_by_input=True))
        sal_A_right = normalize_01(aggregate_saliency(xA_right, xA_right.grad, multiply_by_input=True))
        sal_N_left  = normalize_01(aggregate_saliency(xN_left,  xN_left.grad,  multiply_by_input=True))
        sal_N_right = normalize_01(aggregate_saliency(xN_right, xN_right.grad, multiply_by_input=True))
    except Exception as e:
        logging.warning(f"⚠️ Saliency aggregation failed for {combo_name}: {e}")
        return

    # --- 5️⃣ Prepare metadata ---
    # entry_A = data_module.test_set.data.get(anchor_id, None)
    # if entry_A is None:
    #     logging.warning(f"⚠️ Skipping {combo_name}: no entry for anchor {anchor_id} in dataset.")
    #     return

    # --- 6️⃣ Output dirs ---
    fig_dir = f"{main_dir}/figures/contrast_saliency_random"
    arr_dir = fig_dir.replace("/figures/", "/arrays/")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(arr_dir, exist_ok=True)

    # --- 7️⃣ Plot saliency maps ---
    try:
        plot_saliency_separate_windows(
            sal_A_left, sal_A_right,
            seq_entry=seq_entry_A[0],
            exon_id=combo_name,
            out_prefix=f"{fig_dir}/saliency_combo_{combo_name}_exon_{anchor_id}"
        )
        plot_saliency_separate_windows(
            sal_N_left, sal_N_right,
            seq_entry=seq_entry_N[0],
            exon_id=f"{combo_name}_N",
            out_prefix=f"{fig_dir}/saliency_combo_{combo_name}_exon_{negative_id}"
        )
    except Exception as e:
        logging.warning(f"⚠️ Plotting failed for {combo_name}: {e}")
        return

    # --- 8️⃣ Save arrays & metadata ---
    try:
        np.save(f"{arr_dir}/saliency_{combo_name}_{anchor_id}_left.npy",  sal_A_left)
        np.save(f"{arr_dir}/saliency_{combo_name}_{anchor_id}_right.npy", sal_A_right)
        np.save(f"{arr_dir}/saliency_{combo_name}_{negative_id}_left.npy",  sal_N_left)
        np.save(f"{arr_dir}/saliency_{combo_name}_{negative_id}_right.npy", sal_N_right)

        save_saliency_and_entry(
            base_path=arr_dir,
            exon_id=combo_name,
            sal_left=sal_A_left,
            sal_right=sal_A_right,
            seq_entry=entry_A
        )

        logging.info(f"✅ Saved saliency for {combo_name}")
    except Exception as e:
        logging.warning(f"⚠️ Saving arrays failed for {combo_name}: {e}")
        return

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


# Define main directory for code files (e.g., for saving plots)
main_dir = str(CONTRASTIVE_ROOT / "data" / "extra")


# @hydra.main(version_base=None, config_path=get_config_path(), config_name="config.yaml")
@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def main(config: OmegaConf):



    high_sample = ["ENST00000224337.10_6_17"]
    low_sample  = ["ENST00000296411.11_8_11"]
    logging.info(f"🎯 Selected high exons: {high_sample}")
    logging.info(f"🎯 Selected low exons:  {low_sample}")
    
    # Register Hydra resolvers
    # OmegaConf.register_new_resolver("contrastive_root", lambda: str(CONTRASTIVE_ROOT))
    OmegaConf.register_new_resolver('eval', eval)
    OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
    OmegaConf.register_new_resolver('min', lambda x, y: min([x, y]))
    OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
    OmegaConf.register_new_resolver('optimal_workers', lambda: get_optimal_num_workers())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_config(config, resolve=True)
   
    data_module = ContrastiveIntronsDataModule(config)
    data_module.prepare_data()
    data_module.setup()

    # train_loader = data_module.train_dataloader()
    model = load_pretrained_model(config, device)

    import pickle
    import random

    # path = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/extra/filtered_HiLow_exons_by_tissue.pkl"
    # with open(path, "rb") as f:
    #     data = pickle.load(f)

    # plot_all_anchors_from_list(config, selected_exons, data_module, model, device)
    fig_dir = f"{CONTRASTIVE_ROOT}/data/extra/figures/contrast_saliency_random"
    timestamp = "time"
    for h in high_sample:
        for l in low_sample:
            compute_and_save_saliency_pair(
                anchor_id=h,
                negative_id=l,
                encoder=model,
                data_module=data_module,
                device=device,
                main_dir=fig_dir,
                timestamp=timestamp
            )

if __name__ == "__main__":
    main()