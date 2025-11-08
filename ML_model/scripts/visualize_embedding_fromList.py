from pathlib import Path
from omegaconf import OmegaConf
import os
import pickle
import numpy as np
import glob

def find_contrastive_root(start: Path = Path(__file__)) -> Path:
    for parent in start.resolve().parents:
        if parent.name == "Contrastive_Learning":
            return parent
    raise RuntimeError("Could not find 'Contrastive_Learning' directory.")

# Set env var *before* hydra loads config
os.environ["CONTRASTIVE_ROOT"] = str(find_contrastive_root())
CONTRASTIVE_ROOT = find_contrastive_root()



import sys
import os
import time
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from omegaconf import OmegaConf
import hydra
import time
from sklearn.cluster import KMeans

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



@torch.no_grad()
def plot_all_anchors_from_list(
    config,
    high_expr_exons,
    low_expr_exons,
    data_module,
    model,
    device,
    save_csv=True,
    normalize=True,
):
    """
    Generates t-SNE, UMAP, and PCA embeddings for high vs low expression exons.
    Saves 2D embeddings, figures, and quantitative metrics in a timestamped folder.
    """

    import time, pickle, json, numpy as np
    import matplotlib.pyplot as plt, seaborn as sns
    import torch
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score, silhouette_score
    from sklearn.linear_model import LogisticRegression

    try:
        import umap
        HAS_UMAP = True
    except ImportError:
        HAS_UMAP = False
        print("⚠️ UMAP not available; skipping UMAP embedding.")

    from src.datasets.utility import get_windows_with_padding

    # --- Setup directories ---
    timestamp = time.strftime("%Y_%m_%d__%H_%M_%S")
    out_dir = os.path.join(main_dir, f"figures/tsne_highLowExpr_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # --- Gather embeddings ---
    exon_list = high_expr_exons + low_expr_exons
    dataset = data_module.train_dataloader().dataset
    tokenizer = data_module.tokenizer

    all_embeddings, all_groups = [], []

    for exon_name in exon_list:
        if exon_name not in dataset.data:
            print(f"⚠️ Skipping {exon_name}: not found in dataset.")
            continue

        expr_group = "High" if exon_name in high_expr_exons else "Low"
        seqs = dataset.data[exon_name].get("hg38", None)
        if seqs is None or not all(k in seqs for k in ["5p", "exon", "3p"]):
            continue

        full_seq = seqs["5p"] + seqs["exon"] + seqs["3p"]
        windows = get_windows_with_padding(
            dataset.tissue_acceptor_intron,
            dataset.tissue_donor_intron,
            dataset.tissue_acceptor_exon,
            dataset.tissue_donor_exon,
            full_seq,
            overhang=(dataset.len_3p, dataset.len_5p),
        )

        seql = tokenizer([windows["acceptor"]]).float()
        seqr = tokenizer([windows["donor"]]).float()
        emb = model(seql.to(device), seqr.to(device)).cpu().numpy().squeeze()

        all_embeddings.append(emb)
        all_groups.append(expr_group)

    if len(all_embeddings) < 5:
        print("❌ Not enough embeddings to plot.")
        return

    embeddings = np.vstack(all_embeddings)
    y_bin = np.array([1 if g == "High" else 0 for g in all_groups])

    # --- Normalize embeddings ---
    if normalize:
        embeddings = StandardScaler().fit_transform(embeddings)

    # --- Quantitative metrics on raw embeddings ---
    clf = LogisticRegression(max_iter=2000).fit(embeddings, y_bin)
    auc = roc_auc_score(y_bin, clf.predict_proba(embeddings)[:, 1])
    sil = silhouette_score(embeddings, y_bin)
    metrics = {"AUC_linear": float(auc), "Silhouette": float(sil)}

    # --- Helper: embed + save ---
    def reduce_and_plot(method_name, coords, cmap):
        """Save both figure and coordinates for each embedding method."""
        df_out = np.column_stack([coords, y_bin])
        np.savetxt(
            os.path.join(out_dir, f"{method_name}_embedding.csv"),
            df_out, delimiter=",", header="x,y,label(1=High,0=Low)", comments=""
        )

        plt.figure(figsize=(8, 7))
        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=["High" if y == 1 else "Low" for y in y_bin],
            palette=cmap,
            alpha=0.7, s=35, edgecolor="none",
        )
        sns.kdeplot(x=coords[:, 0], y=coords[:, 1],
                    hue=["High" if y == 1 else "Low" for y in y_bin],
                    levels=3, alpha=0.25, linewidths=1)
        plt.title(f"{method_name}: High vs Low Expression\nAUC={auc:.2f}, Sil={sil:.2f}")
        plt.axis("off")
        plt.tight_layout()
        fig_path = os.path.join(out_dir, f"{method_name.lower()}_plot.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved {method_name} figure and embedding → {fig_path}")

    # --- PCA ---
    pca = PCA(n_components=2)
    coords_pca = pca.fit_transform(embeddings)
    reduce_and_plot("PCA", coords_pca, {"High": "#1f77b4", "Low": "#ff7f0e"})

    # --- t-SNE ---
    tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings) - 1), random_state=42)
    coords_tsne = tsne.fit_transform(embeddings)
    reduce_and_plot("tSNE", coords_tsne, {"High": "#1f77b4", "Low": "#ff7f0e"})

    # --- UMAP ---
    if HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=20, min_dist=0.3, metric="cosine", random_state=42)
        coords_umap = reducer.fit_transform(embeddings)
        reduce_and_plot("UMAP", coords_umap, {"High": "#1f77b4", "Low": "#ff7f0e"})

    # --- Save metrics metadata ---
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"📊 Saved metrics to {out_dir}/metrics.json")



@torch.no_grad()
def plot_all_anchors_from_arrays(
    high_embeddings,
    low_embeddings,
    high_names,
    low_names,
    save_dir,
    normalize=True,
):
    """
    Plot PCA, t-SNE, and UMAP projections directly from KMeans-sampled embeddings.
    No model, tokenizer, or dataset access required.
    """

    import time, json
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import roc_auc_score, silhouette_score
    from sklearn.linear_model import LogisticRegression

    try:
        import umap
        HAS_UMAP = True
    except ImportError:
        HAS_UMAP = False
        print("⚠️ UMAP not available; skipping UMAP embedding.")

    # --- Combine high + low ---
    embeddings = np.vstack([high_embeddings, low_embeddings])
    y_bin = np.concatenate([
        np.ones(len(high_embeddings)),   # High = 1
        np.zeros(len(low_embeddings))    # Low = 0
    ])
    names = np.array(high_names + low_names)

    timestamp = time.strftime("%Y_%m_%d__%H_%M_%S")
    out_dir = os.path.join(save_dir, f"figures/tsne_highLowExpr_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"📊 Plotting {len(embeddings)} embeddings (dim={embeddings.shape[1]})")

    # --- Normalize ---
    if normalize:
        embeddings = StandardScaler().fit_transform(embeddings)

    # --- Metrics on raw embeddings ---
    clf = LogisticRegression(max_iter=2000).fit(embeddings, y_bin)
    auc = roc_auc_score(y_bin, clf.predict_proba(embeddings)[:, 1])
    sil = silhouette_score(embeddings, y_bin)
    metrics = {"AUC_linear": float(auc), "Silhouette": float(sil)}

    # --- Helper to plot & save ---
    def reduce_and_plot(name, coords):
        np.savetxt(
            os.path.join(out_dir, f"{name}_embedding.csv"),
            np.column_stack([coords, y_bin]),
            delimiter=",", header="x,y,label(1=High,0=Low)", comments=""
        )

        plt.figure(figsize=(8, 7))
        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=["High" if y == 1 else "Low" for y in y_bin],
            palette={"High": "#1f77b4", "Low": "#ff7f0e"},
            alpha=0.75, s=35, edgecolor="none",
        )
        sns.kdeplot(
            x=coords[:, 0], y=coords[:, 1],
            hue=["High" if y == 1 else "Low" for y in y_bin],
            levels=3, alpha=0.25, linewidths=1,
        )
        plt.title(f"{name}: High vs Low Expression\nAUC={auc:.2f}, Sil={sil:.2f}")
        plt.axis("off")
        fig_path = os.path.join(out_dir, f"{name.lower()}_plot.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved {name} figure → {fig_path}")

    # --- PCA ---
    # coords_pca = PCA(n_components=2).fit_transform(embeddings)
    # reduce_and_plot("PCA", coords_pca)

    # # --- t-SNE ---
    # tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings)-1), random_state=42)
    # coords_tsne = tsne.fit_transform(embeddings)
    # reduce_and_plot("tSNE", coords_tsne)

    # --- UMAP ---
    # if HAS_UMAP:
    #     reducer = umap.UMAP(n_neighbors=20, min_dist=0.3, metric="cosine", random_state=42)
    #     coords_umap = reducer.fit_transform(embeddings)
    #     reduce_and_plot("UMAP", coords_umap)

    if HAS_UMAP:
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



    # --- Save metrics ---
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"📈 Saved metrics → {out_dir}/metrics.json")



@torch.no_grad()
def extract_and_save_exon_embeddings(
    config,
    high_expr_exons,
    low_expr_exons,
    data_module,
    model,
    device,
    main_dir=None,
    save_combined_npz=True,
):
    """
    Extracts embeddings for a list of high/low expression exons using the given model,
    saves individual exon .pkl files, and optionally saves a combined .npz archive.

    Returns:
        all_embeddings (np.ndarray): Combined embeddings
        all_exons (List[str]): Corresponding exon names
        all_groups (List[str]): 'High' or 'Low'
    """
    import time, pickle
    import numpy as np
    from src.datasets.utility import get_windows_with_padding

    if main_dir is None:
        main_dir = os.path.join(str(CONTRASTIVE_ROOT), "data/extra")

    # --- Create a timestamped folder ---
    timestamp = time.strftime("%Y_%m_%d__%H_%M_%S")
    out_dir = os.path.join(main_dir, f"tsne_embeddings_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    dataset = data_module.train_dataloader().dataset
    tokenizer = data_module.tokenizer

    exon_list = high_expr_exons + low_expr_exons

    all_embeddings, all_exons, all_groups = [], [], []

    print(f"🧬 Extracting embeddings for {len(exon_list)} exons...")

    for exon_name in exon_list:
        if exon_name not in dataset.data:
            print(f"⚠️ Skipping {exon_name}: not found in dataset.")
            continue

        expr_group = "High" if exon_name in high_expr_exons else "Low"
        all_views_dict = dataset.data[exon_name]

        # --- Only human for now ---
        species_names = ["hg38"]

        exon_embeddings = []
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
                overhang=(dataset.len_3p, dataset.len_5p),
            )

            seql = tokenizer([windows["acceptor"]]).float()
            seqr = tokenizer([windows["donor"]]).float()
            emb = model(seql.to(device), seqr.to(device)).cpu().numpy()

            exon_embeddings.append(emb.squeeze())

        if len(exon_embeddings) == 0:
            continue

        exon_embeddings = np.vstack(exon_embeddings)

        # --- Save individual exon embedding ---
        exon_out = os.path.join(out_dir, f"{exon_name}_hg38_{expr_group}_embedding.pkl")
        with open(exon_out, "wb") as f:
            pickle.dump(
                {
                    "exon_name": exon_name,
                    "group": expr_group,
                    "embeddings": exon_embeddings,
                },
                f,
            )
        print(f"💾 Saved: {os.path.basename(exon_out)}")

        # --- Append to global lists ---
    #     all_embeddings.extend(exon_embeddings)
    #     all_exons.extend([exon_name] * len(exon_embeddings))
    #     all_groups.extend([expr_group] * len(exon_embeddings))

    # all_embeddings = np.vstack(all_embeddings)
    # print(f"\n✅ Finished embedding extraction: {len(all_embeddings)} total embeddings.")

    # --- Optionally save a combined npz file ---
    if save_combined_npz:
        npz_path = os.path.join(out_dir, f"combined_embeddings_{timestamp}.npz")
        np.savez_compressed(
            npz_path,
            embeddings=all_embeddings,
            exons=np.array(all_exons),
            groups=np.array(all_groups),
        )
        print(f"📦 Saved combined embeddings → {npz_path}")

    return all_embeddings, all_exons, all_groups



# @torch.no_grad()
# def plot_all_anchors_from_list(config, high_expr_exons, low_expr_exons, data_module, model, device, save_csv=True):
#     """
#     Follows EXACT same flow as training (Dataset -> Collate -> Tokenizer -> Encoder)
#     and plots all exons + homologs together in ONE t-SNE figure,
#     grouping exons into High vs Low expression sets.
#     """

#     import os
#     import time
#     import pickle
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     from sklearn.manifold import TSNE
#     from src.datasets.utility import get_windows_with_padding

#     # --- Define high and low expression exons ---
#     exon_list = high_expr_exons + low_expr_exons

#     dataset = data_module.train_dataloader().dataset
#     tokenizer = data_module.tokenizer
#     os.makedirs(main_dir, exist_ok=True)

#     all_embeddings = []
#     all_species = []
#     all_exons = []
#     all_groups = []  # NEW: expression-level labels

#     for exon_name in exon_list:
#         if exon_name not in dataset.data:
#             print(f"⚠️ Skipping {exon_name}: not found in dataset.")
#             continue

#         expr_group = "High" if exon_name in high_expr_exons else "Low"

#         all_views_dict = dataset.data[exon_name]
#         # species_names = list(all_views_dict.keys())
#         # species_names = ['hg38', 'mm10', 'bosTau8', 'rn6', 'xenTro7', 'galGal4', 'danRer10']
#         species_names = ['hg38']

#         exon_embeddings = []
#         exon_species = []

#         for sp in species_names:
#             seqs = all_views_dict[sp]
#             if not all(k in seqs for k in ["5p", "exon", "3p"]):
#                 print(f"⚠️ Skipping {sp} for {exon_name}: missing 5p/3p/exon keys")
#                 continue

#             full_seq = seqs["5p"] + seqs["exon"] + seqs["3p"]
#             windows = get_windows_with_padding(
#                 dataset.tissue_acceptor_intron,
#                 dataset.tissue_donor_intron,
#                 dataset.tissue_acceptor_exon,
#                 dataset.tissue_donor_exon,
#                 full_seq,
#                 overhang=(dataset.len_3p, dataset.len_5p)
#             )

#             seql = tokenizer([windows["acceptor"]]).float()
#             seqr = tokenizer([windows["donor"]]).float()
#             with torch.no_grad():
#                 emb = model(seql.to(device), seqr.to(device)).cpu().numpy()

#             exon_embeddings.append(emb.squeeze())
#             exon_species.append(sp)

#         if len(exon_embeddings) == 0:
#             print(f"⚠️ No valid homologs for {exon_name}, skipping.")
#             continue

#         exon_embeddings = np.vstack(exon_embeddings)

#         # --- Save individual exon embeddings ---
#         out_path = os.path.join(str(CONTRASTIVE_ROOT), f"tsne_embedding/{exon_name}_hg38_{expr_group}_embedding.pkl")
#         os.makedirs(os.path.dirname(out_path), exist_ok=True)
#         with open(out_path, "wb") as f:
#             pickle.dump(
#                 {"exon_name": exon_name, "species": exon_species, "embeddings": exon_embeddings},
#                 f
#             )
#         print(f"✅ Saved embeddings for {exon_name}: {out_path}")
        
#         # Append to global collections
#         all_embeddings.extend(exon_embeddings)
#         all_exons.extend([exon_name] * len(exon_embeddings))
#         all_species.extend(exon_species)
#         all_groups.extend([expr_group] * len(exon_embeddings))

#     if len(all_embeddings) < 2:
#         print("❌ Not enough embeddings to plot.")
#         return

#     embeddings = np.vstack(all_embeddings)
#     tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings) - 1), random_state=42)
#     emb_2d = tsne.fit_transform(embeddings)

#     plt.figure(figsize=(9, 8))
#     ax = sns.scatterplot(
#         x=emb_2d[:, 0],
#         y=emb_2d[:, 1],
#         hue=all_groups,        # color by expression level (High/Low)
#         style=all_species,     # shape by species
#         s=70,
#         alpha=0.9,
#         palette={"High": "#1f77b4", "Low": "#ff7f0e"},
#         edgecolor="black",
#         linewidth=0.3,
#         legend="brief"
#     )

#     # --- Remove species legend (keep only High/Low) ---
#     handles, labels = ax.get_legend_handles_labels()
#     keep = [i for i, label in enumerate(labels) if label in ["High", "Low"]]
#     ax.legend([handles[i] for i in keep], [labels[i] for i in keep], title="Expression")

#     plt.title(f"t-SNE: High vs Low Expression Exons\n({len(exon_list)} exons, {len(embeddings)} embeddings)")
#     plt.axis("off")
#     plt.tight_layout()

#     out_path = os.path.join(main_dir, f"figures/tsne_highLowExpr_{time.strftime('%Y_%m_%d__%H_%M_%S')}.png")
#     plt.savefig(out_path, dpi=300, bbox_inches="tight")
#     plt.close()
#     print(f"✅ Saved combined t-SNE plot: {out_path}")
    



# Define main directory for code files (e.g., for saving plots)
main_dir = str(CONTRASTIVE_ROOT / "data" / "extra")


# @hydra.main(version_base=None, config_path=get_config_path(), config_name="config.yaml")
@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def main(config: OmegaConf):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==============================================================
    #  Choose how to sample
    # ==============================================================

    N_HIGH = 1000
    N_LOW  = 500
    SAMPLE_METHOD = "kmeans"   # or "random"
      # ==============================================================
    #  Option 1: Random uniform sampling
    # ==============================================================

    if SAMPLE_METHOD == "random":

        # Register Hydra resolvers
        # OmegaConf.register_new_resolver("contrastive_root", lambda: str(CONTRASTIVE_ROOT))
        OmegaConf.register_new_resolver('eval', eval)
        OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
        OmegaConf.register_new_resolver('min', lambda x, y: min([x, y]))
        OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
        OmegaConf.register_new_resolver('optimal_workers', lambda: get_optimal_num_workers())

        print_config(config, resolve=True)
    
        data_module = ContrastiveIntronsDataModule(config)
        data_module.prepare_data()
        data_module.setup()

        train_loader = data_module.train_dataloader()
        model = load_pretrained_model(config, device)
        
        # --- Aggregate all high/low expression exons across tissues ---
        path = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/extra/filtered_HiLow_exons_by_tissue.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        high_expr_exons, low_expr_exons = [], []

        for tissue, values in data.items():
            high_list = values.get("high_expression_exons", [])
            low_list = values.get("low_expression_exons", [])
            if not high_list and not low_list:
                continue  # skip empty tissues

            print(f"{tissue:<40} | High: {len(high_list):4d} | Low: {len(low_list):4d}")
            high_expr_exons.extend(high_list)
            low_expr_exons.extend(low_list)

        print(f"\n✅ Total high-expression exons: {len(high_expr_exons):,}")
        print(f"✅ Total low-expression exons:  {len(low_expr_exons):,}")

        sampled_high = random.sample(high_expr_exons, min(N_HIGH, len(high_expr_exons)))
        sampled_low  = random.sample(low_expr_exons,  min(N_LOW, len(low_expr_exons)))

        # ==============================================================
        #  Summary + log file
        # ==============================================================
        print(f"\n🎯 Selected {len(sampled_high)} high-expression exons")
        print(f"🎯 Selected {len(sampled_low)} low-expression exons")

        # Save the sampled exon lists for reproducibility
        import time
        timestamp = time.strftime("%Y_%m_%d__%H_%M_%S")
        sample_dir = os.path.join(main_dir, f"sampled_exons_{timestamp}")
        os.makedirs(sample_dir, exist_ok=True)

        with open(os.path.join(sample_dir, "high_exons.txt"), "w") as f:
            f.write("\n".join(sampled_high))
        with open(os.path.join(sample_dir, "low_exons.txt"), "w") as f:
            f.write("\n".join(sampled_low))

        print(f"🗂️  Saved sampled exon lists → {sample_dir}/")

        # ==============================================================
        #  Plot the selected exons
        # ==============================================================
        plot_all_anchors_from_list(config, sampled_high, sampled_low, data_module, model, device)


    # ==============================================================
    #  Option 2: Diversity-based sampling (KMeans centroids)
    # ==============================================================
    elif SAMPLE_METHOD == "kmeans":
        import glob, pickle

        embedding_dir = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/extra/tsne_embeddings_2025_11_07__19_57_20" # small amount of exons, for testing
        print(f"📂 Loading embeddings from: {embedding_dir}")

        high_embeddings, low_embeddings = [], []
        high_names, low_names = [], []

        for fpath in glob.glob(os.path.join(embedding_dir, "*_embedding.pkl")):
            try:
                with open(fpath, "rb") as f:
                    data = pickle.load(f)
                exon_name = data.get("exon_name", os.path.basename(fpath).split("_hg38")[0])
                emb = np.array(data["embeddings"]).squeeze()
                group = data.get("group", "Unknown").lower()

                if group == "high":
                    high_embeddings.append(emb)
                    high_names.append(exon_name)
                elif group == "low":
                    low_embeddings.append(emb)
                    low_names.append(exon_name)
            except Exception as e:
                print(f"⚠️ Skipping {fpath}: {e}")

        all_high_embeddings = np.vstack(high_embeddings)
        all_low_embeddings  = np.vstack(low_embeddings)

        print(f"✅ Loaded {len(all_high_embeddings)} high and {len(all_low_embeddings)} low exon embeddings.")
        print(f"   Each embedding has dimension {all_high_embeddings.shape[1]}.\n")

        # --- Cluster diversity sampling ---
        def cluster_sample(embeddings, exon_names, n_clusters=1000):
            n_clusters = min(n_clusters, len(embeddings))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            centers = kmeans.cluster_centers_

            chosen = []
            for i in range(n_clusters):
                cluster_idx = np.where(labels == i)[0]
                if len(cluster_idx) == 0:
                    continue
                dists = np.linalg.norm(embeddings[cluster_idx] - centers[i], axis=1)
                exemplar = cluster_idx[np.argmin(dists)]
                chosen.append(exon_names[exemplar])
            return chosen

        sampled_high = cluster_sample(all_high_embeddings, high_names, n_clusters=N_HIGH)
        sampled_low  = cluster_sample(all_low_embeddings,  low_names,  n_clusters=N_LOW)
        # Step 3: Collect embeddings for those sampled exons
        mask_high = [high_names.index(e) for e in sampled_high]
        mask_low  = [low_names.index(e) for e in sampled_low]
        sampled_high_embeddings = all_high_embeddings[mask_high]
        sampled_low_embeddings  = all_low_embeddings[mask_low]

        # Step 4: Plot only those
        plot_all_anchors_from_arrays(
            high_embeddings=sampled_high_embeddings,
            low_embeddings=sampled_low_embeddings,
            high_names=sampled_high,
            low_names=sampled_low,
            save_dir=main_dir,
        )


    else:
        raise ValueError(f"Unknown SAMPLE_METHOD = {SAMPLE_METHOD}")




    """

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

    train_loader = data_module.train_dataloader()
    model = load_pretrained_model(config, device)
    
    # --- Load the .pkl dict ---
    import pickle
    import random

    path = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/extra/filtered_HiLow_exons_by_tissue.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)

    # --- Aggregate all high/low expression exons across tissues ---
    high_expr_exons, low_expr_exons = [], []

    for tissue, values in data.items():
        high_list = values.get("high_expression_exons", [])
        low_list = values.get("low_expression_exons", [])
        if not high_list and not low_list:
            continue  # skip empty tissues

        print(f"{tissue:<40} | High: {len(high_list):3d} | Low: {len(low_list):3d}")
        high_expr_exons.extend(high_list)
        low_expr_exons.extend(low_list)

    print(f"\n✅ Total high-expression exons: {len(high_expr_exons)}")
    print(f"✅ Total low-expression exons:  {len(low_expr_exons)}")

    # --- Randomly select 10 from each group ---
    # n_high = min(1, len(high_expr_exons))
    # n_low = min(1, len(low_expr_exons))
    n_high = max(1, len(high_expr_exons))
    n_low = max(1, len(low_expr_exons))
    # n_high = 500
    # n_low = 500

    random_high = random.sample(high_expr_exons, n_high)
    random_low = random.sample(low_expr_exons, n_low)

    print(f"\n🎯 Selected {n_high} random high-expression exons:")
    # print(random_high)
    print(f"\n🎯 Selected {n_low} random low-expression exons:")

    embeddings, exons, groups = extract_and_save_exon_embeddings(
    config,
    random_high, random_low,
    data_module, model, device,
    main_dir="/gpfs/commons/home/atalukder/Contrastive_Learning/data/extra")

        
    """



if __name__ == "__main__":
    main()
