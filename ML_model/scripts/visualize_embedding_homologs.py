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





@torch.no_grad()
def plot_all_anchors_from_list(config, exon_list, data_module, model, device, save_csv=True):
    """
    Follows EXACT same flow as training (Dataset -> Collate -> Tokenizer -> Encoder)
    and plots all exons + homologs together in ONE t-SNE figure,
    grouping exons into High vs Low expression sets.
    """


    try:
        import umap
        HAS_UMAP = True
    except ImportError:
        HAS_UMAP = False
        print("⚠️ UMAP not available; skipping UMAP embedding.")

    normalize = True  # Whether to normalize embeddings before plotting

    # --- Setup directories ---
    timestamp = time.strftime("%Y_%m_%d__%H_%M_%S")
    out_dir = os.path.join(main_dir, f"figures/tsne_2exon_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)


    # --- Define high and low expression exons ---
    

    dataset = data_module.train_dataloader().dataset
    tokenizer = data_module.tokenizer
    os.makedirs(main_dir, exist_ok=True)

    all_embeddings = []
    all_species = []
    all_exons = []
    all_groups = []  # NEW: expression-level labels
    expr_group = "High"

    for exon_name in exon_list:
        if exon_name not in dataset.data:
            print(f"⚠️ Skipping {exon_name}: not found in dataset.")
            continue

        # expr_group = "High" if exon_name in high_expr_exons else "Low"

        all_views_dict = dataset.data[exon_name]
        species_names = list(all_views_dict.keys())

        exon_embeddings = []
        exon_species = []

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
            with torch.no_grad():
                emb = model(seql.to(device), seqr.to(device)).cpu().numpy()

            exon_embeddings.append(emb.squeeze())
            exon_species.append(sp)

        if len(exon_embeddings) == 0:
            print(f"⚠️ No valid homologs for {exon_name}, skipping.")
            continue

        exon_embeddings = np.vstack(exon_embeddings)

        # --- Save individual exon embeddings ---
        out_path = os.path.join(str(CONTRASTIVE_ROOT), f"data/extra/tsne_embedding/{exon_name}_embedding.pkl")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(
                {"exon_name": exon_name, "species": exon_species, "embeddings": exon_embeddings},
                f
            )
        print(f"✅ Saved embeddings for {exon_name}: {out_path}")

        # Append to global collections
        all_embeddings.extend(exon_embeddings)
        all_exons.extend([exon_name] * len(exon_embeddings))
        all_species.extend(exon_species)
        all_groups.extend([expr_group] * len(exon_embeddings))

    if len(all_embeddings) < 2:
        print("❌ Not enough embeddings to plot.")
        return
    

    embeddings = np.vstack(all_embeddings)

        # --- UMAP ---
        # --- UMAP ---
    if HAS_UMAP:
        params = [
            (5, 0.1), (10, 0.1), (15, 0.2), (30, 0.3), (50, 0.5)
        ]

        print(f"🌀 Running UMAP sweep ({len(params)} parameter sets)...")

        for n_nb, m_dist in params:
            name = f"UMAP_n{n_nb}_d{m_dist}"
            print(f"🔧 Computing {name} ...")

            # reducer = umap.UMAP(
            #     n_neighbors=n_nb,
            #     min_dist=m_dist,
            #     metric="cosine",
            #     random_state=42
            # )
            # coords = reducer.fit_transform(embeddings)

            reducer = joblib.load("/gpfs/commons/home/atalukder/Contrastive_Learning/data/extra/figures/tsne_highLowExpr_2025_11_11__16_57_16/UMAP_n10_d0.1_model.pkl")
            coords = reducer.transform(np.vstack(all_embeddings))

            # --- Save reducer object ---
            model_path = os.path.join(out_dir, f"{name}_model.pkl")
            joblib.dump(reducer, model_path)
            print(f"💾 Saved fitted UMAP model → {model_path}")

            # --- Save CSV with exon + species info ---
            df_out = pd.DataFrame({
                "exon_name": all_exons,
                "species": all_species,
                "group": all_groups,
                "x": coords[:, 0],
                "y": coords[:, 1]
            })
            csv_path = os.path.join(out_dir, f"{name}_embedding.csv")
            df_out.to_csv(csv_path, index=False)
            print(f"💾 Saved embedding CSV → {csv_path}")

            # --- Color by exon ---
            unique_exons = list(df_out["exon_name"].unique())
            palette = sns.color_palette("tab10", n_colors=len(unique_exons))

            plt.figure(figsize=(8, 7))
            sns.scatterplot(
                data=df_out,
                x="x", y="y",
                hue="exon_name",
                palette=palette,
                s=50, edgecolor="black", linewidth=0.3, alpha=0.9
            )
            plt.title(f"{name}: Homolog Embeddings (colored by exon)")
            plt.legend(
                title="Exon", 
                bbox_to_anchor=(1.05, 1), loc='upper left',
                borderaxespad=0., fontsize=8, title_fontsize=9
            )
            plt.axis("off")

            fig_path = os.path.join(out_dir, f"{name.lower()}_coloredByExon.png")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"✅ Saved {name} plot colored by exon → {fig_path}")

        print(f"📊 All UMAP variants, models, and plots saved under: {out_dir}")



    # y_bin = np.array([1 if g == "High" else 0 for g in all_groups])

    # # --- Normalize embeddings ---
    # if normalize:
    #     embeddings = StandardScaler().fit_transform(embeddings)

    # # --- Quantitative metrics on raw embeddings ---
    # clf = LogisticRegression(max_iter=2000).fit(embeddings, y_bin)
    # auc = roc_auc_score(y_bin, clf.predict_proba(embeddings)[:, 1])
    # sil = silhouette_score(embeddings, y_bin)
    # metrics = {"AUC_linear": float(auc), "Silhouette": float(sil)}

    # # --- Helper: embed + save ---
    # def reduce_and_plot(method_name, coords, cmap):
    #     """Save both figure and coordinates for each embedding method."""
    #     df_out = np.column_stack([coords, y_bin])
    #     np.savetxt(
    #         os.path.join(out_dir, f"{method_name}_embedding.csv"),
    #         df_out, delimiter=",", header="x,y,label(1=High,0=Low)", comments=""
    #     )

    #     plt.figure(figsize=(8, 7))
    #     sns.scatterplot(
    #         x=coords[:, 0],
    #         y=coords[:, 1],
    #         hue=["High" if y == 1 else "Low" for y in y_bin],
    #         palette=cmap,
    #         alpha=0.7, s=35, edgecolor="none",
    #     )
    #     sns.kdeplot(x=coords[:, 0], y=coords[:, 1],
    #                 hue=["High" if y == 1 else "Low" for y in y_bin],
    #                 levels=3, alpha=0.25, linewidths=1)
    #     plt.title(f"{method_name}: High vs Low Expression\nAUC={auc:.2f}, Sil={sil:.2f}")
    #     plt.axis("off")
    #     plt.tight_layout()
    #     fig_path = os.path.join(out_dir, f"{method_name.lower()}_plot.png")
    #     plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    #     plt.close()
    #     print(f"✅ Saved {method_name} figure and embedding → {fig_path}")

    # # --- PCA ---
    # pca = PCA(n_components=2)
    # coords_pca = pca.fit_transform(embeddings)
    # reduce_and_plot("PCA", coords_pca, {"High": "#1f77b4", "Low": "#ff7f0e"})

    # # --- t-SNE ---
    # tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings) - 1), random_state=42)
    # coords_tsne = tsne.fit_transform(embeddings)
    # reduce_and_plot("tSNE", coords_tsne, {"High": "#1f77b4", "Low": "#ff7f0e"})

    # # --- UMAP ---
    # if HAS_UMAP:
    #     params = [
    #         (5, 0.1), (10, 0.1), (15, 0.2), (30, 0.3), (50, 0.5)
    #     ]

    #     print(f"🌀 Running UMAP sweep ({len(params)} parameter sets)...")

    #     for n_nb, m_dist in params:
    #         name = f"UMAP_n{n_nb}_d{m_dist}"
    #         print(f"🔧 Computing {name} ...")

    #         reducer = umap.UMAP(
    #             n_neighbors=n_nb,
    #             min_dist=m_dist,
    #             metric="cosine",
    #             random_state=42
    #         )
    #         coords = reducer.fit_transform(embeddings)
    #         reduce_and_plot(name, coords, {"High": "#1f77b4", "Low": "#ff7f0e"})

    #     print(f"📊 All UMAP variants saved under: {out_dir}")

    # # --- Save metrics metadata ---
    # with open(os.path.join(out_dir, "metrics.json"), "w") as f:
    #     json.dump(metrics, f, indent=2)
    # print(f"📊 Saved metrics to {out_dir}/metrics.json")




import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

def get_selected_exons():

    root_dir = '/gpfs/commons/home/atalukder/Contrastive_Learning/data/extra/figures/'
    csv_path = f"{root_dir}/tsne_highLowExpr_2025_11_11__16_57_16/UMAP_n10_d0.1_embeddingWid.csv"

    # --- Load CSV ---
    df = pd.read_csv(csv_path)
    coords = df[['x','y']].values
    labels = df['label(1=High,0=Low)'].values
    names  = df['exon_name'].values  # ✅ This is what you were missing

    # --- Split by class ---
    high_idx = np.where(labels == 1)[0]
    low_idx  = np.where(labels == 0)[0]

    # --- Pick the two farthest points within each class ---
    def farthest_two(coords, idx):
        sub = coords[idx]
        d = cdist(sub, sub)
        i, j = np.unravel_index(np.argmax(d), d.shape)
        return [idx[i], idx[j]]

    sel_high = farthest_two(coords, high_idx)
    sel_low  = farthest_two(coords, low_idx)
    selected_idx = sel_high + sel_low

    # --- Retrieve exon names ---
    selected_exons = [names[i] for i in selected_idx]

    print("Selected 4 exons (2 High, 2 Low):")
    for exon in selected_exons:
        print("  -", exon)

    return selected_exons


# Define main directory for code files (e.g., for saving plots)
main_dir = str(CONTRASTIVE_ROOT / "data" / "extra")


# @hydra.main(version_base=None, config_path=get_config_path(), config_name="config.yaml")
@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def main(config: OmegaConf):



    selected_exons = get_selected_exons()
    
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

    plot_all_anchors_from_list(config, selected_exons, data_module, model, device)


if __name__ == "__main__":
    main()