from pathlib import Path
from omegaconf import OmegaConf
import os, sys, time, json, logging, numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import hydra
import seaborn as sns
import os, sys, random, logging, torch
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
import hydra
import time

timestamp = time.strftime("_%Y_%m_%d__%H_%M_%S")

# ---------------- root discovery ----------------
from pathlib import Path
def find_contrastive_root(start: Path = Path(__file__)) -> Path:
    for parent in start.resolve().parents:
        if parent.name == "Contrastive_Learning":
            return parent
    raise RuntimeError("Could not find 'Contrastive_Learning' directory.")

root_path = str(find_contrastive_root())
os.environ["CONTRASTIVE_ROOT"] = root_path
print(f"CONTRASTIVE_ROOT set to: {root_path}")

# ---------------- imports from your repo ----------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.config import print_config
from src.datasets.auxiliary_jobs import PSIRegressionDataModule
from src.utils.encoder_init import initialize_encoders_and_model   # <-- we will only take the encoder features

# ---------------- user overrides ----------------
ANCHOR_EXON_ID   = os.environ.get("ANCHOR_EXON_ID",  "")
NEGATIVE_EXON_ID = os.environ.get("NEGATIVE_EXON_ID","")
TISSUE_NAME      = os.environ.get("TISSUE_NAME",     "lung")  # not strictly needed here, but kept for metadata
SIM_TYPE         = os.environ.get("SIM_TYPE", "cosine")       # "cosine" or "dot"
RESULT_DIR       = os.environ.get("RESULT_DIR", "exprmnt_2025_11_01__13_07_04") # "exprmnt_2025_10_25__15_31_32"
comment = "EMPRAICL_afterSweep_aug10_200bp_2025_11_01__13_07_04, with AG...GT"

# timestamp = time.strftime("_%Y_%m_%d__%H_%M_%S")
# main_dir = f"{root_path}/code/ML_model"
# os.makedirs(f"{main_dir}/figures", exist_ok=True)
# os.makedirs(f"{main_dir}/arrays",  exist_ok=True)

# ---------------- logging ----------------
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


import os
import numpy as np

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


# ---------------- helpers: similarity, saliency, plotting ----------------
def compute_similarity(zA: torch.Tensor, zN: torch.Tensor, sim_type: str = "cosine") -> torch.Tensor:
    if sim_type == "cosine":
        return F.cosine_similarity(zA, zN, dim=-1).mean()
    elif sim_type == "dot":
        return (zA * zN).sum(dim=-1).mean()
    raise ValueError("SIM_TYPE must be 'cosine' or 'dot'.")

def get_exon_batch_by_id(data_module, exon_id: str, loader_type: str = "train"):
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

    for batch in loader:
        # batch: ((seql, seqr), psi_vals, exon_ids)
        if isinstance(batch, (tuple, list)) and len(batch) == 3:
            (seql, seqr), psi_vals, exon_ids = batch
        else:
            raise RuntimeError("Unexpected batch format from PSIRegressionDataModule.")

        exon_ids = list(exon_ids)
        if exon_id in exon_ids:
            idx = exon_ids.index(exon_id)
            logging.info(f"✅ Found exon_id {exon_id} in batch.")
            # Each batch is: ((seql, seqr), psi_vals, exon_ids)
            return (seql, seqr, psi_vals, exon_ids), idx

    raise ValueError(f"Exon ID {exon_id} not found in any batch.")


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

def plot_saliency(s, title, out_png):
    plt.figure(figsize=(10, 2.2))
    plt.plot(s, linewidth=1.4)
    plt.title(title); plt.xlabel("Position"); plt.ylabel("Saliency")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

# ---------------- NEW: dataset-aware sequence fetch & one-hot ----------------
_DNA2IDX = {"A":0, "C":1, "G":2, "T":3}
def dna_one_hot(seq: str) -> torch.Tensor:
    """
    Returns tensor of shape (4, L) channels-first.
    Unknowns (N / others) -> all zeros (grad will just not emphasize them).
    """
    seq = seq.upper()
    L = len(seq)
    x = torch.zeros(4, L, dtype=torch.float32)
    for i, ch in enumerate(seq):
        j = _DNA2IDX.get(ch, None)
        if j is not None:
            x[j, i] = 1.0
    return x


import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

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
    exon_start_seq = seq_entry["exon"].get("start", "")
    exon_end_seq = seq_entry["exon"].get("end", "")

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
        tissue_acceptor_intron - len_5p_real :
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
    plt.savefig(f"{out_prefix}_both.png", dpi=200)
    plt.close()

    print(f"[✅] Saved seql and seqr saliency plots for {exon_id}")



import numpy as np
import matplotlib.pyplot as plt


def fetch_lr_tensors_from_dataset(dataset, exon_id: str, device: torch.device):
    """
    Your PSIRegressionDataModule test dataset (per screenshot) stores:
      dataset.data[exon_id] = { 'psi_val': float, '3p': str, '5p': str, 'exon': {...} }
    We will:
      - read '5p' and '3p' strings
      - one-hot encode to (1, 4, L)
      - mark requires_grad_(True)
    """
    if not exon_id:
        raise RuntimeError("Missing exon_id.")
    if not hasattr(dataset, "data"):
        raise RuntimeError("Dataset has no `.data` mapping.")
    if exon_id not in dataset.data:
        raise RuntimeError(f"Exon id {exon_id} not found in dataset.data keys.")
    entry = dataset.data[exon_id]

    # keys could be '5p', '3p' (as in screenshot). Be tolerant to variants:
    s5 = entry.get("5p") or entry.get("fivep") or entry.get("left") or entry.get("seql")
    s3 = entry.get("3p") or entry.get("threep") or entry.get("right") or entry.get("seqr")
    if (s5 is None) or (s3 is None):
        raise RuntimeError(f"Could not find 5p/3p sequences for {exon_id}. Keys: {list(entry.keys())}")

    x5 = dna_one_hot(s5).unsqueeze(0).to(device)  # (1,4,L)
    x3 = dna_one_hot(s3).unsqueeze(0).to(device)  # (1,4,L)
    x5.requires_grad_(True); x3.requires_grad_(True)
    return x5, x3


def compute_and_save_saliency_pair(anchor_id, negative_id, encoder, data_module, device, main_dir, timestamp):
    """Compute saliency between two exons, skipping gracefully if one is missing."""
    combo_name = f"{anchor_id}_vs_{negative_id}"

    # --- 1️⃣ Try fetching both batches safely ---
    try:
        (anchor_batch, iA) = get_exon_batch_by_id(data_module, anchor_id, loader_type="test")
    except Exception as e:
        logging.warning(f"⚠️ Skipping pair {combo_name}: could not fetch anchor exon {anchor_id} — {e}")
        return

    try:
        (negative_batch, iN) = get_exon_batch_by_id(data_module, negative_id, loader_type="test")
    except Exception as e:
        logging.warning(f"⚠️ Skipping pair {combo_name}: could not fetch negative exon {negative_id} — {e}")
        return

    try:
        (seql_A, seqr_A, psiA, exon_ids_A) = anchor_batch
        (seql_N, seqr_N, psiN, exon_ids_N) = negative_batch
    except Exception as e:
        logging.warning(f"⚠️ Invalid batch structure for {combo_name}: {e}")
        return

    # --- 2️⃣ Build tensors ---
    try:
        xA_left  = seql_A[iA].unsqueeze(0).float().to(device).requires_grad_(True)
        xA_right = seqr_A[iA].unsqueeze(0).float().to(device).requires_grad_(True)
        xN_left  = seql_N[iN].unsqueeze(0).float().to(device).requires_grad_(True)
        xN_right = seqr_N[iN].unsqueeze(0).float().to(device).requires_grad_(True)
        
    except Exception as e:
        logging.warning(f"⚠️ Tensor creation failed for {combo_name}: {e}")
        return

    # --- 3️⃣ Encoder forward & backward ---
    try:
        zA = encoder(xA_left, xA_right)
        zN = encoder(xN_left, xN_right)
        sim = compute_similarity(zA, zN, SIM_TYPE)

        encoder.zero_grad(set_to_none=True)
        for t in (xA_left, xA_right, xN_left, xN_right):
            if t.grad is not None:
                t.grad.zero_()
        (-sim).backward()
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
    entry_A = data_module.test_set.data.get(anchor_id, None)
    if entry_A is None:
        logging.warning(f"⚠️ Skipping {combo_name}: no entry for anchor {anchor_id} in dataset.")
        return
    entry_N = data_module.test_set.data.get(negative_id, None)
    if entry_N is None:
        logging.warning(f"⚠️ Skipping {combo_name}: no entry for negative {negative_id} in dataset.")
        return

    # --- 6️⃣ Output dirs ---
    
    fig_dir = f"{main_dir}/figures"
    arr_dir = f"{main_dir}/arrays"
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(arr_dir, exist_ok=True)

    # --- 7️⃣ Plot saliency maps ---
    try:
        plot_saliency_separate_windows(
            sal_A_left, sal_A_right,
            seq_entry=entry_A,
            exon_id=combo_name,
            out_prefix=f"{fig_dir}/saliency_combo_{combo_name}_exon_{anchor_id}"
        )
        plot_saliency_separate_windows(
            sal_N_left, sal_N_right,
            seq_entry=entry_N,
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

        # save_saliency_and_entry(
        #     base_path=arr_dir,
        #     exon_id=combo_name,
        #     sal_left=sal_A_left,
        #     sal_right=sal_A_right,
        #     seq_entry=entry_A
        # )

        logging.info(f"✅ Saved saliency for {combo_name} | PSI(A)={psiA[iA].item():.3f}, PSI(N)={psiN[iN].item():.3f}")
    except Exception as e:
        logging.warning(f"⚠️ Saving arrays failed for {combo_name}: {e}")
        return



# ---------------- HYDRA MAIN ----------------
@hydra.main(version_base=None, config_path="../configs", config_name="psi_regression.yaml")
def main(config: OmegaConf):
    # logging

    out_dir = f"{root_path}/data/extra/contrast_saliency_random/f_{timestamp}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)

    # resolvers
    OmegaConf.register_new_resolver('eval', eval)
    OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
    OmegaConf.register_new_resolver('min', lambda x, y: min([x, y]))
    OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
    OmegaConf.register_new_resolver('optimal_workers', lambda: min(os.cpu_count() // max(1, torch.cuda.device_count()), 8))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- configure PSI dataset (you were already doing this) ---
    OmegaConf.set_struct(config, False)
    # Point to your PSI pkl for the chosen tissue
    # (You had Retina/Eye here; keep or override via Hydra/ENV as you prefer)
    config.dataset.test_files.intronexon = f"{root_path}/data/final_data/ASCOT_finetuning/psi_variable_Retina___Eye_psi_MERGED.pkl"
    config.dataset.fivep_ovrhang = 200
    config.dataset.threep_ovrhang = 200
    config.aux_models.mtsplice_weights = RESULT_DIR
    config.aux_models.warm_start = True
    config.aux_models.mode = "mtsplice"
    config.task.global_batch_size = 2048
    OmegaConf.set_struct(config, True)

    print_config(config, resolve=True)
    logging.info(f"🎯 CL_weight: {RESULT_DIR}")
    logging.info(f"{comment}")

    # --- Data ---
    data_module = PSIRegressionDataModule(config)
    data_module.setup()
    logging.info(f"Dataset size (test): {len(data_module.test_set)}; keys available: {len(getattr(data_module.test_set,'data',{}))}")
 
    encoder_full = initialize_encoders_and_model(config, root_path)
    encoder = encoder_full.encoder  # 🔥 take only encoder submodule
    encoder = encoder.to(device).float().eval()

    # --- Load high/low exons CSV ---
    common_csv = f"{root_path}/data/extra/common_high_low_exons.csv"
    common_df = pd.read_csv(common_csv).dropna(how="all")
    high_exons = common_df["common_high_accuracy_high"].dropna().tolist()
    low_exons  = common_df["common_high_accuracy_low"].dropna().tolist()

    # --- Randomly select 3 high and 3 low ---
    # random.seed(42)
    # high_sample = random.sample(high_exons, min(20, len(high_exons)))
    # low_sample  = random.sample(low_exons,  min(20, len(low_exons)))

    high_sample = ['GT_62125', 'GT_34667', 'GT_05391', 'GT_24765', 'GT_30881', 'GT_53818', 'GT_42813', 'GT_56990', 'GT_25156', 'GT_77786', 'GT_43914', 'GT_28682', 'GT_04544', 'GT_11538', 'GT_70831', 'GT_14195', 'GT_33825', 'GT_73433', 'GT_19230', 'GT_02570']
    low_sample  = ['GT_09591', 'GT_58515', 'GT_53963', 'GT_71001', 'GT_45119', 'GT_20109', 'GT_21527', 'GT_06314', 'GT_19466', 'GT_25036', 'GT_79749', 'GT_29902', 'GT_38954', 'GT_43067', 'GT_68350', 'GT_05383', 'GT_53279', 'GT_76431', 'GT_27762', 'GT_57800']


    # --- Randomly select 3 high and 3 low ---
    # random.seed(42)
    # high_sample = ['GT_03764']
    # low_sample  = ['GT_23158']
    logging.info(f"🎯 Selected high exons: {high_sample}")
    logging.info(f"🎯 Selected low exons:  {low_sample}")


    # saliency_combo__vs_GT_23158_exon_GT_03764_both
   
    # --- Perform all 3×3 combinations ---
    
    for h in high_sample:
        for l in low_sample:
            compute_and_save_saliency_pair(
                anchor_id=h,
                negative_id=l,
                encoder=encoder,
                data_module=data_module,
                device=device,
                main_dir=out_dir,
                timestamp=timestamp
            )

    logging.info("✅ Completed all contrastive saliency combinations.")



if __name__ == "__main__":
    main()