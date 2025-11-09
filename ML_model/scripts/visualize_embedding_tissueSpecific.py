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
RESULT_DIR       = os.environ.get("RESULT_DIR", "exprmnt_2025_10_25__15_31_32") # 200 bp "exprmnt_2025_10_25__15_31_32" # 300 bp  "exprmnt_2025_10_26__14_29_04"

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
            # logging.info(f"✅ Found exon_id {exon_id} in batch.")
            # Each batch is: ((seql, seqr), psi_vals, exon_ids)
            return (seql, seqr, psi_vals, exon_ids), idx

    raise ValueError(f"Exon ID {exon_id} not found in any batch.")


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
import time
timestamp = time.strftime("%Y_%m_%d__%H_%M_%S")

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_high_low_per_tissue(
    df_embeds,
    out_dir="./plots_tissuewise_highlow",
    normalize=True,
    run_pca=False,
    run_tsne=False,
    run_umap=True,
):
    """
    Visualize high vs low embeddings *within each tissue*.

    Args:
        df_embeds (pd.DataFrame): Must contain columns [embedding, tissue, label].
        out_dir (str): Output folder for the plots.
        normalize (bool): Standardize embeddings before projection.
        run_pca, run_tsne, run_umap (bool): Dimensionality reduction options.
    """
    import umap
    out_dir = os.path.join(out_dir, f"perTissue_embeddings_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    tissues = sorted(df_embeds["tissue"].unique())
    print(f"🧬 Found {len(tissues)} tissues: {tissues}")

    for tissue in tissues:
        subset = df_embeds[df_embeds["tissue"] == tissue]
        if subset["label"].nunique() < 2:
            print(f"⚠️ Skipping {tissue} — only one label present.")
            continue

        X = np.vstack(subset["embedding"])
        y = subset["label"].to_numpy()

        if normalize:
            X = StandardScaler().fit_transform(X)

        # ---------------------------------------------
        # Dimensionality reduction (choose PCA/t-SNE/UMAP)
        # ---------------------------------------------
        if run_pca:
            reducer_name = "PCA"
            reducer = PCA(n_components=2, random_state=42)
        elif run_tsne:
            reducer_name = "tSNE"
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
        elif run_umap:
            reducer_name = "UMAP"
            reducer = umap.UMAP(n_neighbors=20, min_dist=0.3, metric="cosine", random_state=42)
        else:
            raise ValueError("At least one reduction method must be True.")

        coords = reducer.fit_transform(X)

        # ---------------------------------------------
        # Plot
        # ---------------------------------------------
        plt.figure(figsize=(7, 6))
        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=y,
            s=60,
            alpha=0.9,
            palette={"high": "#E74C3C", "low": "#3498DB"},
            edgecolor="none"
        )
        plt.title(f"{reducer_name}: {tissue} (High vs Low)")
        plt.axis("off")
        plt.legend(title="Expression Label", loc="best", fontsize=9)
        plt.tight_layout()

        safe_tissue = (
            tissue.replace("/", "_")
                .replace("\\", "_")
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace(",", "")
        )
        save_path = os.path.join(out_dir, f"{safe_tissue}_{reducer_name}.png")
        # save_path = os.path.join(out_dir, f"{tissue.replace(' ', '_')}_{reducer_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved {tissue} → {save_path}")


@torch.no_grad()
def visualize_embeddings_single_label(
    embeddings: np.ndarray,
    tissues: np.ndarray,
    label: str,
    out_dir: str = "./embedding_plots",
    normalize: bool = True,
    run_pca: bool = True,
    run_tsne: bool = True,
    run_umap: bool = True,
    save_csv: bool = True,
):
    """
    Visualize exon embeddings for a single label (High or Low) colored by tissue.

    Args:
        embeddings (np.ndarray): Array of shape (N, D) — embeddings for this label only.
        tissues (np.ndarray or list): Array of tissue names (length N).
        label (str): Expression label ("High" or "Low").
        out_dir (str): Output directory for plots.
        normalize (bool): Standardize embeddings before reduction.
        run_pca, run_tsne, run_umap (bool): Dimensionality reduction options.
        save_csv (bool): Save reduced coordinates as CSV.
    """
    import os, time
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    try:
        import umap
        HAS_UMAP = True
    except ImportError:
        HAS_UMAP = False
        print("⚠️ UMAP not available; skipping UMAP embedding.")
        run_umap = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
   
    out_dir = os.path.join(out_dir, f"{label}_embeddings_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"📂 Visualizing {len(embeddings)} {label}-expression embeddings across {len(np.unique(tissues))} tissues.")

    if normalize:
        embeddings = StandardScaler().fit_transform(embeddings)

    # ------------------------------------------------------------------
    # Helper plotting function
    # ------------------------------------------------------------------
    def reduce_and_plot(method_name, coords):
        df = np.column_stack([coords, tissues])
        if save_csv:
            np.savetxt(
                os.path.join(out_dir, f"{method_name}_embedding.csv"),
                df,
                fmt="%s",
                delimiter=",",
                header="x,y,tissue",
                comments=""
            )

        plt.figure(figsize=(9, 7))
        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=tissues,
            s=55,
            alpha=0.9,
            edgecolor="none",
            # palette="tab20"
            palette={"high": "#d62728", "low": "#1f77b4"}
        )
        plt.title(f"{method_name}: {label}-expression exon embeddings")
        plt.axis("off")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=1, fontsize=9)
        plt.tight_layout()
        fig_path = os.path.join(out_dir, f"{method_name}_{label}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved {method_name} plot → {fig_path}")

    # ------------------------------------------------------------------
    # Run dimensionality reductions
    # ------------------------------------------------------------------
    if run_pca:
        pca = PCA(n_components=2, random_state=42)
        coords_pca = pca.fit_transform(embeddings)
        reduce_and_plot("PCA", coords_pca)

    if run_tsne:
        tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings) - 1), random_state=42)
        coords_tsne = tsne.fit_transform(embeddings)
        reduce_and_plot("tSNE", coords_tsne)

    # if run_umap and HAS_UMAP:
    #     reducer = umap.UMAP(n_neighbors=20, min_dist=0.3, metric="cosine", random_state=42)
    #     coords_umap = reducer.fit_transform(embeddings)
    #     reduce_and_plot("UMAP", coords_umap)

    if run_umap and HAS_UMAP:
        params = [
            (5, 0.1), (10, 0.1), (15, 0.2), (30, 0.3), (50, 0.5)
        ]

        print(f"🌀 Running UMAP sweep ({len(params)} parameter sets)...")

        for n_nb, m_dist in params:
            name = f"UMAP_n{n_nb}_d{m_dist}"
            print(f"🔧 Computing {name} ...")

            reducer = umap.UMAP(
                n_neighbors=n_nb,
                min_dist=m_dist,
                metric="cosine",
                random_state=42
            )
            coords = reducer.fit_transform(embeddings)
            reduce_and_plot(name, coords)

        print(f"📊 All UMAP variants saved under: {out_dir}")




    print(f"📊 All {label} visualizations saved under: {out_dir}")


def get_embedding(anchor_id, encoder, data_module, device, main_dir, timestamp):
    """Compute saliency between two exons, skipping gracefully if one is missing."""


    # --- 1️⃣ Try fetching both batches safely ---
    try:
        (anchor_batch, iA) = get_exon_batch_by_id(data_module, anchor_id, loader_type="test")
    except Exception as e:
        logging.warning(f"⚠️ Skipping pair {anchor_id}: could not fetch anchor exon {anchor_id} — {e}")
        return

    try:
        (seql_A, seqr_A, psiA, exon_ids_A) = anchor_batch
    except Exception as e:
        logging.warning(f"⚠️ Invalid batch structure for {anchor_id}: {e}")
        return

    # --- 2️⃣ Build tensors ---
    try:
        xA_left  = seql_A[iA].unsqueeze(0).float().to(device).requires_grad_(True)
        xA_right = seqr_A[iA].unsqueeze(0).float().to(device).requires_grad_(True)
    except Exception as e:
        logging.warning(f"⚠️ Tensor creation failed for {anchor_id}: {e}")
        return

    # --- 3️⃣ Encoder forward & backward ---
    try:
        zA = encoder(xA_left, xA_right)
        return zA.detach().cpu().numpy()
    except Exception as e:
        logging.warning(f"⚠️ Encoder forward/backward failed for {anchor_id}: {e}")
        return



# ---------------- HYDRA MAIN ----------------
"""
 # (AT) do not erase
###############################
@hydra.main(version_base=None, config_path="../configs", config_name="psi_regression.yaml")
def main(config: OmegaConf):
# def main():

    # logging
    out_dir = Path(f"{root_path}/data/extra/figures")
    setup_logging(out_dir)

    # (AT) Do not erase
    
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

    # --- Data ---
    data_module = PSIRegressionDataModule(config)
    data_module.setup()
    logging.info(f"Dataset size (test): {len(data_module.test_set)}; keys available: {len(getattr(data_module.test_set,'data',{}))}")

 
    encoder_full = initialize_encoders_and_model(config, root_path)
    encoder = encoder_full.encoder  # 🔥 take only encoder submodule
    encoder = encoder.to(device).float().eval()


    # ==========================================================
    # 🔍 Load ASCOT tissue-specific binary PSI table
    # ==========================================================
    ascot_csv = f"{root_path}/data/ASCOT/variable_cassette_exons_with_binary_labels_ExonBinPsi.csv"
    ascot_df = pd.read_csv(ascot_csv)
    logging.info(f"📖 Loaded ASCOT exon PSI table: {ascot_df.shape[0]} exons × {ascot_df.shape[1]} columns")

    # Get unique exon IDs
    unique_exons = ascot_df["exon_id"].dropna().unique().tolist()
    logging.info(f"✅ Found {len(unique_exons):,} unique exons.")

    # ============================================================
    # ⚙️ Step 2. Generate embeddings for all exons
    # ============================================================
    all_embeddings, all_exon_ids = [], []

    for i, exon_id in enumerate(unique_exons):
        if i % 100 == 0:
            logging.info(f"Progress: {i:,}/{len(unique_exons):,} exons...")

        zA = get_embedding(
            anchor_id=exon_id,
            encoder=encoder,
            data_module=data_module,
            device=device,
            main_dir=f"{root_path}/code/ML_model/figures/contrast_saliency_random",
            timestamp=timestamp
        )

        if zA is not None:
            all_embeddings.append(zA.squeeze())
            all_exon_ids.append(exon_id)
            # print(f"✅ {exon_id} done")
        else:
            logging.warning(f"⚠️ Skipped exon {exon_id} — not found or encoding failed.")

    logging.info(f"✅ Encoded {len(all_embeddings):,} / {len(unique_exons):,} exons successfully.")

    # ============================================================
    # 💾 Step 3. Save all exon embeddings
    # ============================================================
    if len(all_embeddings) > 0:
        emb_array = np.vstack(all_embeddings)
        out_path = f"{root_path}/data/extra/all_ASCOT_Variable_exon_embeddings_{timestamp}.npz"
        np.savez_compressed(
            out_path,
            embeddings=emb_array,
            exon_ids=np.array(all_exon_ids, dtype=object)
        )
        logging.info(f"💾 Saved all exon embeddings → {out_path}")
    else:
        logging.warning("⚠️ No embeddings were generated.")
"""
        
# (AT) do not erase
###############################

def main():

    # logging
    out_dir = Path(f"{root_path}/data/extra/figures")
    setup_logging(out_dir)
    
    ascot_csv = f"{root_path}/data/ASCOT/variable_cassette_exons_with_binary_labels_ExonBinPsi.csv"
    ascot_df = pd.read_csv(ascot_csv)
    logging.info(f"📖 Loaded ASCOT exon PSI table: {ascot_df.shape[0]} exons × {ascot_df.shape[1]} columns")


    # Known metadata columns (the first few before tissue-specific columns)
    meta_cols = [
        "exon_id", "cassette_exon", "alternative_splice_site_group",
        "linked_exons", "mutually_exclusive_exons", "exon_strand",
        "exon_length", "gene_type", "gene_id", "gene_symbol",
        "exon_location", "exon_biotype", " 'chromosome.1'"
    ]
    tissue_cols = [c for c in ascot_df.columns if c not in meta_cols]
    logging.info(f"🧬 Detected {len(tissue_cols)} tissue columns.")

    # --- Count number of 1s and 0s per tissue ---
    tissue_counts = []
    for t in tissue_cols:
        n_high = (ascot_df[t] == 1).sum()
        n_low  = (ascot_df[t] == 0).sum()
        tissue_counts.append((t, n_high, n_low))

    # Sort by total number of labeled exons (1+0)
    tissue_counts.sort(key=lambda x: x[1]+x[2], reverse=True)

    # --- Select top 5 tissues with most 1/0 annotations ---
    # top5_tissues = [t[0] for t in tissue_counts[:3]]
    top5_tissues = [t[0] for t in tissue_counts]
    print(f"🔥 Top 5 tissues selected: {top5_tissues}")

    # --- Prepare high/low exon sets per tissue ---
    tissue_to_exons = {"high": {}, "low": {}}

    for tissue in top5_tissues:
        # Exons labeled 1 or 0 for this tissue
        high_exons = ascot_df.loc[ascot_df[tissue] == 1, "exon_id"].dropna().unique().tolist()
        low_exons  = ascot_df.loc[ascot_df[tissue] == 0, "exon_id"].dropna().unique().tolist()

        tissue_to_exons["high"][tissue] = list(set(high_exons))
        tissue_to_exons["low"][tissue]  = list(set(low_exons))

    print(f"✅ Extracted unique exon IDs per tissue for both high/low sets")

    data = np.load(f"{root_path}/data/extra/all_ASCOT_Variable_exon_embeddings__2025_11_07__22_56_27.npz", allow_pickle=True) # "exprmnt_2025_10_26__14_29_04"
    # data = np.load(f"{root_path}/data/extra/all_ASCOT_Variable_exon_embeddings_2025_11_08__00_41_13.npz", allow_pickle=True) # "exprmnt_2025_10_25__15_31_32"
    embeddings = data["embeddings"]     # shape (N, D)
    exon_ids   = data["exon_ids"]       # shape (N,)
    embeddings_dict = {eid: emb for eid, emb in zip(exon_ids, embeddings)}

    # -------------------------------------------------------------------
    # 🧠 for each tissue plots high vs low
    # -------------------------------------------------------------------

    # (AT) do not erase
    ###############################
#     tissue_to_unique = {"high": {t: [] for t in top5_tissues},
#                     "low":  {t: [] for t in top5_tissues}}

#     seen = {"high": set(), "low": set()}

#     for label in ["high", "low"]:
#         for tissue in top5_tissues:
#             uniq = []
#             for ex in tissue_to_exons[label][tissue]:
#                 if ex not in seen[label]:
#                     uniq.append(ex)
#                     seen[label].add(ex)
#             tissue_to_unique[label][tissue] = uniq


#     # --- Gather embeddings for each tissue and label ---
#     data_points = []

#     for label, tissue_dict in tissue_to_unique.items():
#         for tissue, exon_list in tissue_dict.items():
#             for exon in exon_list:
#                 if exon in embeddings_dict:
#                     emb = embeddings_dict[exon]
#                     data_points.append({
#                         "exon_id": exon,
#                         "tissue": tissue,
#                         "label": label,
#                         "embedding": emb
#                     })

#     print(f"💾 Collected {len(data_points)} exon–tissue–label entries with embeddings")
#     # --- Convert to DataFrame (for plotting, PCA, etc.) ---
#     embed_dim = len(data_points[0]["embedding"])
#     embed_array = np.vstack([d["embedding"] for d in data_points])
#     # df_embeds = pd.DataFrame(embed_array, columns=[f"dim_{i}" for i in range(embed_dim)])
#     # df_embeds["tissue"] = [d["tissue"] for d in data_points]
#     # df_embeds["label"]  = [d["label"]  for d in data_points]
#     # df_embeds["exon_id"] = [d["exon_id"] for d in data_points]
#     df_embeds = pd.DataFrame({
#     "embedding": [d["embedding"] for d in data_points],
#     "tissue": [d["tissue"] for d in data_points],
#     "label": [d["label"] for d in data_points],
#     "exon_id": [d["exon_id"] for d in data_points],
# })

#     # embed_cols = [c for c in df_embeds.columns if c.startswith("dim_")]


#     # Suppose df_embeds is your combined DataFrame
#     mask_high = df_embeds["label"] == "high"
#     mask_low  = df_embeds["label"] == "low"


#     visualize_embeddings_single_label(
#         embeddings=np.vstack(df_embeds.loc[mask_high, "embedding"]),
#         tissues=df_embeds.loc[mask_high, "tissue"].to_numpy(),
#         label="High",
#         out_dir=out_dir
#     )

#     visualize_embeddings_single_label(
#         embeddings=np.vstack(df_embeds.loc[mask_low, "embedding"]),
#         tissues=df_embeds.loc[mask_low, "tissue"].to_numpy(),
#         label="Low",
#         out_dir=out_dir
#     )

    # (AT) do not erase
    ###############################

    
#     # --- Gather embeddings for each tissue and label ---
#     data_points = []
#     tissue_to_unique = tissue_to_exons
#     for label, tissue_dict in tissue_to_unique.items():
#         for tissue, exon_list in tissue_dict.items():
#             for exon in exon_list:
#                 if exon in embeddings_dict:
#                     emb = embeddings_dict[exon]
#                     data_points.append({
#                         "exon_id": exon,
#                         "tissue": tissue,
#                         "label": label,
#                         "embedding": emb
#                     })

#     print(f"💾 Collected {len(data_points)} exon–tissue–label entries with embeddings")
#     # --- Convert to DataFrame (for plotting, PCA, etc.) ---
#     df_embeds = pd.DataFrame({
#     "embedding": [d["embedding"] for d in data_points],
#     "tissue": [d["tissue"] for d in data_points],
#     "label": [d["label"] for d in data_points],
#     "exon_id": [d["exon_id"] for d in data_points],
# })


#     visualize_high_low_per_tissue(
#     df_embeds=df_embeds,
#     out_dir=out_dir,
#     normalize=True,
#     run_pca=False,   # You can also try tSNE or UMAP later
#     run_tsne=False,
#     run_umap=True,
# )
    
    # (AT) do not erase
    ###############################


    # -------------------------------------------------------------------
    # 🧠 Combine all brain-related tissues into a single "Brain_All"
    # -------------------------------------------------------------------
    brain_tissues = [
        "Cerebellar Hemisphere - Brain",
        "Amygdala - Brain",
        "Anterior cingulate - Brain",
        "Cerebellum - Brain",
        "Cortex - Brain",
        "Frontal Cortex - Brain",
        "Hippocampus - Brain",
        "Hypothalamus - Brain",
        "Nucleus accumbens - Brain",
        "Putamen - Brain",
        "Spinal cord C1 - Brain",
        "Substantia nigra - Brain",
    ]

    # --- Merge all high/low exons across brain tissues ---
    combined_high, combined_low = set(), set()
    for bt in brain_tissues:
        combined_high.update(tissue_to_exons["high"].get(bt, []))
        combined_low.update(tissue_to_exons["low"].get(bt, []))

    # --- Remove overlapping exons (high & low in brain) ---
    overlap = combined_high & combined_low
    print(f"🧠 Combined {len(brain_tissues)} brain tissues")
    print(f"   High exons before filter: {len(combined_high)}")
    print(f"   Low exons before filter:  {len(combined_low)}")
    print(f"   Overlap (to remove):       {len(overlap)} ({len(overlap)/(len(combined_high|combined_low)):.2%})")

    combined_high -= overlap
    combined_low  -= overlap

    print(f"✅ After removing overlaps → High: {len(combined_high)}, Low: {len(combined_low)}")


    # --- Build embedding lookup ---
    data = np.load(f"{root_path}/data/extra/all_ASCOT_Variable_exon_embeddings_2025_11_08__00_41_13.npz", allow_pickle=True)
    embeddings = data["embeddings"]
    exon_ids   = data["exon_ids"]
    embeddings_dict = {eid: emb for eid, emb in zip(exon_ids, embeddings)}

    # --- Gather cleaned brain embeddings only ---
    data_points = []
    for exon in combined_high:
        if exon in embeddings_dict:
            data_points.append({
                "exon_id": exon,
                "tissue": "Brain_All",
                "label": "high",
                "embedding": embeddings_dict[exon],
            })
    for exon in combined_low:
        if exon in embeddings_dict:
            data_points.append({
                "exon_id": exon,
                "tissue": "Brain_All",
                "label": "low",
                "embedding": embeddings_dict[exon],
            })

    print(f"💾 Collected {len(data_points)} cleaned Brain_All exon–label entries")

    # --- Convert to DataFrame for visualization ---
    df_embeds = pd.DataFrame({
        "embedding": [d["embedding"] for d in data_points],
        "tissue": [d["tissue"] for d in data_points],
        "label": [d["label"] for d in data_points],
        "exon_id": [d["exon_id"] for d in data_points],
    })

    # --- Visualize High vs Low (Brain_All only) ---
    visualize_embeddings_single_label(
    embeddings=np.vstack(df_embeds["embedding"]),
    tissues=df_embeds["label"].to_numpy(),   # ✅ use labels for coloring
    label="Brain_All",
    out_dir=out_dir,
    normalize=True,
    run_pca=False,
    run_tsne=False,
    run_umap=True,
)


    


if __name__ == "__main__":
    main()
