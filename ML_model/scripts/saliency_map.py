from pathlib import Path
from omegaconf import OmegaConf
import os, sys, time, json, logging, numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import hydra
import seaborn as sns

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
RESULT_DIR       = os.environ.get("RESULT_DIR", "exprmnt_2025_10_25__15_31_32")

timestamp = time.strftime("_%Y_%m_%d__%H_%M_%S")
main_dir = f"{root_path}/code/ML_model"
os.makedirs(f"{main_dir}/figures", exist_ok=True)
os.makedirs(f"{main_dir}/arrays",  exist_ok=True)

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

# def plot_saliency_dynamic_barplot(
#     sal_left: np.ndarray,
#     sal_right: np.ndarray,
#     seq_entry: dict,
#     exon_id: str,
#     out_png: str,
#     crop_threshold: float = 0.02,
#     tissue_acceptor_intron: float = 200,
#     tissue_donor_exon: float = 100,
# ):
#     """
#     Plot saliency as a bar plot across real exon/intron layout derived from seq_entry.

#     Args:
#         sal_left:  np.ndarray of saliency for acceptor (5′ side)
#         sal_right: np.ndarray of saliency for donor (3′ side)
#         seq_entry: dict from data_module.test_set.data[exon_id]
#                    e.g. {"5p": "...", "3p": "...", "exon": {"start": "...", "end": "..."}}
#         exon_id:   ID for title
#         out_png:   Output filename
#         crop_threshold: cutoff to trim very low saliency regions
#     """

#     # --- 1️⃣ Extract real sequences ---
#     seq_5p = seq_entry["5p"]
#     seq_3p = seq_entry["3p"]
#     exon_start_seq = seq_entry["exon"].get("start", "")
#     exon_end_seq = seq_entry["exon"].get("end", "")

#     # Remove artificial Ns used for padding
#     seq_5p_real = seq_5p.rstrip("N")
#     seq_3p_real = seq_3p.lstrip("N")
#     exon_start_real = exon_start_seq.rstrip("N")
#     exon_end_real = exon_end_seq.lstrip("N")

#     len_5p_real = len(seq_5p_real)
#     len_3p_real = len(seq_3p_real)
#     len_exon_start_real = len(exon_start_real)
#     len_exon_end_real = len(exon_end_real)

#     # --- 2️⃣ Biological slicing of saliency maps ---
#     sal_left_trimmed = sal_left[
#         tissue_acceptor_intron - len_5p_real :
#         tissue_acceptor_intron + len_exon_start_real
#     ]
#     sal_right_trimmed = sal_right[
#         tissue_acceptor_intron+tissue_donor_exon - len_exon_end_real :
#         tissue_acceptor_intron+tissue_donor_exon + len_3p_real
#     ]

#     combined_sal = np.concatenate([sal_left_trimmed, sal_right_trimmed])

#     combined_sal = (combined_sal - combined_sal.min()) / (combined_sal.max() - combined_sal.min() + 1e-8)

#     total_len = len(combined_sal)
#     positions = np.arange(total_len)

#     # --- 4️⃣ Mark exon boundaries ---
#     exon_start_idx = tissue_acceptor_intron
#     exon_end_idx = exon_start_idx + len_exon_start_real + len_exon_end_real

#     # --- 5️⃣ Plot ---
#     plt.figure(figsize=(12, 3))
#     plt.bar(positions, combined_sal, color="dimgray", width=1.0, linewidth=0)

#     plt.axvspan(0, exon_start_idx, color="lightblue", alpha=0.3, label="5′ Intron")
#     plt.axvspan(exon_start_idx, exon_end_idx, color="lightgreen", alpha=0.3, label="Exon")
#     plt.axvspan(exon_end_idx, total_len, color="lightcoral", alpha=0.3, label="3′ Intron")

#     plt.axvline(exon_start_idx, color="k", linestyle="--", linewidth=0.8)
#     plt.axvline(exon_end_idx, color="k", linestyle="--", linewidth=0.8)

#     plt.title(f"Saliency across exon {exon_id}", fontsize=13)
#     plt.xlabel("Sequence position (5′→3′)", fontsize=11)
#     plt.ylabel("Normalized saliency", fontsize=11)
#     plt.legend(frameon=False, loc="upper right")
#     plt.tight_layout()
#     plt.savefig(out_png, dpi=200)
#     plt.close()

#     print(f"[✅] Saved biological saliency for {exon_id}: {out_png}")


import numpy as np
import matplotlib.pyplot as plt

def plot_saliency_dynamic_barplot(
    sal_left: np.ndarray,
    sal_right: np.ndarray,
    exon_id: str,
    out_png: str,
    tissue_acceptor_intron: int = 300,
    tissue_donor_exon: int = 100,
):
    """
    Plot saliency as a single continuous bar plot:
    [0–200]   → 5′ intron
    [200–400] → exon
    [400–600] → 3′ intron

    Args:
        sal_left:  np.ndarray of saliency for acceptor (5′ side)
        sal_right: np.ndarray of saliency for donor (3′ side)
        exon_id:   ID for title
        out_png:   Output filename
    """

    # --- 1️⃣ Normalize saliency ---
    sal_left = (sal_left - sal_left.min()) / (sal_left.max() - sal_left.min() + 1e-8)
    sal_right = (sal_right - sal_right.min()) / (sal_right.max() - sal_right.min() + 1e-8)

    # --- 2️⃣ Truncate to expected biological window sizes ---
    # sal_left = sal_left[:tissue_acceptor_intron + tissue_donor_exon]   # 5′ intron + exon start
    # sal_right = sal_right[:tissue_donor_exon + tissue_acceptor_intron]    # exon end + 3′ intron

    # --- 3️⃣ Concatenate ---
    combined_sal = np.concatenate([sal_left, sal_right])
    total_len = len(combined_sal)
    positions = np.arange(total_len)

    # --- 4️⃣ Define boundaries ---
    exon_start_idx = tissue_acceptor_intron
    exon_end_idx = exon_start_idx + tissue_donor_exon * 2  # start+end = 200 bp exon
    intron_end_idx = exon_end_idx + tissue_acceptor_intron

    # --- 5️⃣ Plot ---
    plt.figure(figsize=(12, 3))
    plt.bar(positions, combined_sal, color="dimgray", width=1.0, linewidth=0)

    # Visual annotation of regions
    plt.axvspan(0, exon_start_idx, color="lightblue", alpha=0.3, label="5′ Intron")
    plt.axvspan(exon_start_idx, exon_end_idx, color="lightgreen", alpha=0.3, label="Exon")
    plt.axvspan(exon_end_idx, intron_end_idx, color="lightcoral", alpha=0.3, label="3′ Intron")

    plt.axvline(exon_start_idx, color="k", linestyle="--", linewidth=0.8)
    plt.axvline(exon_end_idx, color="k", linestyle="--", linewidth=0.8)

    plt.title(f"Saliency across exon {exon_id}", fontsize=13)
    plt.xlabel("Position (5′→3′)", fontsize=11)
    plt.ylabel("Normalized saliency", fontsize=11)
    plt.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"[✅] Saved simplified saliency map for {exon_id}: {out_png}")



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

# ---------------- HYDRA MAIN ----------------
@hydra.main(version_base=None, config_path="../configs", config_name="psi_regression.yaml")
def main(config: OmegaConf):
    # logging
    out_dir = Path(f"{root_path}/files/results/{RESULT_DIR}/contrastive_saliency")
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
    config.dataset.test_files.intronexon = f"{root_path}/data/final_data_old/ASCOT_finetuning/psi_variable_Retina___Eye_psi_MERGED.pkl"
    config.dataset.fivep_ovrhang = 200
    config.dataset.threep_ovrhang = 200
    config.aux_models.mtsplice_weights = RESULT_DIR
    config.aux_models.warm_start = True
    config.aux_models.mode = "mtsplice"
    config.task.global_batch_size = 2048
    OmegaConf.set_struct(config, True)

    print_config(config, resolve=True)

    # --- Data ---
    data_module = PSIRegressionDataModule(config)
    data_module.setup()
    logging.info(f"Dataset size (test): {len(data_module.test_set)}; keys available: {len(getattr(data_module.test_set,'data',{}))}")

    ANCHOR_EXON_ID = "GT_44593"   # your given exon id
    NEGATIVE_EXON_ID = "GT_66306" # your given exon id
    # --- Use your existing function EXACTLY as-is to fetch batches containing the IDs ---
    # --- Use your existing function as-is ---
    (anchor_batch, iA) = get_exon_batch_by_id(data_module, ANCHOR_EXON_ID, loader_type="test")
    (negative_batch, iN) = get_exon_batch_by_id(data_module, NEGATIVE_EXON_ID, loader_type="test")

    # Each batch is ((seql, seqr), psi_vals, exon_ids)
    (seql_A, seqr_A, psiA, exon_ids_A) = anchor_batch
    (seql_N, seqr_N, psiN, exon_ids_N) = negative_batch

    # --- Extract just the tokenized tensors for that exon ---
    xA_left  = seql_A[iA].unsqueeze(0).float().to(device).requires_grad_(True)
    xA_right = seqr_A[iA].unsqueeze(0).float().to(device).requires_grad_(True)

    xN_left  = seql_N[iN].unsqueeze(0).float().to(device).requires_grad_(True)
    xN_right = seqr_N[iN].unsqueeze(0).float().to(device).requires_grad_(True)

    print(f"✅ Anchor exon: {exon_ids_A[iA]}, PSI = {psiA[iA].item():.3f}")
    print(f"✅ Negative exon: {exon_ids_N[iN]}, PSI = {psiN[iN].item():.3f}")

 
    encoder_full = initialize_encoders_and_model(config, root_path)
    encoder = encoder_full.encoder  # 🔥 take only encoder submodule
    encoder = encoder.to(device).float().eval()
    
    # --- 3) Encoder-only forward to get embeddings (features) ---
    zA = encoder(xA_left, xA_right)  # shape: (1, D)
    zN = encoder(xN_left, xN_right)  # shape: (1, D)

    # --- 4) Similarity and backprop to inputs ---
    sim = compute_similarity(zA, zN, SIM_TYPE)  # "cosine" or "dot"
    encoder.zero_grad(set_to_none=True)
    for t in (xA_left, xA_right, xN_left, xN_right):
        if t.grad is not None: t.grad.zero_()
    sim.backward()

    # --- 5) Aggregate per-position saliency and save ---
    sal_A_left  = normalize_01(aggregate_saliency(xA_left,  xA_left.grad,  multiply_by_input=True))
    sal_A_right = normalize_01(aggregate_saliency(xA_right, xA_right.grad, multiply_by_input=True))
    sal_N_left  = normalize_01(aggregate_saliency(xN_left,  xN_left.grad,  multiply_by_input=True))
    sal_N_right = normalize_01(aggregate_saliency(xN_right, xN_right.grad, multiply_by_input=True))

    baseA = f"{main_dir}/figures/contrast_saliency_MTSplice_A{timestamp}"
    baseN = f"{main_dir}/figures/contrast_saliency_MTSplice_N{timestamp}"
    # plot_saliency(sal_A_left,  f"Anchor A ({exon_ids_A[iA]}) 5′ window",  baseA+"_left.png")
    # plot_saliency(sal_A_right, f"Anchor A ({exon_ids_A[iA]}) 3′ window",  baseA+"_right.png")
    # plot_saliency(sal_N_left,  f"Negative N ({exon_ids_N[iN]}) 5′ window", baseN+"_left.png")
    # plot_saliency(sal_N_right, f"Negative N ({exon_ids_N[iN]}) 3′ window", baseN+"_right.png")
    entry = data_module.test_set.data[ANCHOR_EXON_ID]
    
    # plot_saliency_dynamic_barplot(
    #     sal_A_left, sal_A_right,
    #     seq_entry=entry,
    #     exon_id="GT_44593",
    #     out_png=f"{main_dir}/figures/saliency_2_{ANCHOR_EXON_ID}.png"
    # )
    plot_saliency_dynamic_barplot(
        sal_A_left, sal_A_right,
        exon_id="GT_44593",
        out_png=f"{main_dir}/figures/saliency_2_{ANCHOR_EXON_ID}.png"
    )
    array_dir = baseA.replace("/figures/", "/arrays/")
    os.makedirs(array_dir, exist_ok=True)

    # np.save(baseA.replace("/figures/","/arrays/")+"_left.npy",  sal_A_left)
    # np.save(baseA.replace("/figures/","/arrays/")+"_right.npy", sal_A_right)
    # np.save(baseN.replace("/figures/","/arrays/")+"_left.npy",  sal_N_left)
    # np.save(baseN.replace("/figures/","/arrays/")+"_right.npy", sal_N_right)

    logging.info(f"✅ Contrastive saliency done:\n  A: {exon_ids_A[iA]} (PSI {psiA[iA].item():.3f})\n  N: {exon_ids_N[iN]} (PSI {psiN[iN].item():.3f})")


if __name__ == "__main__":
    main()
