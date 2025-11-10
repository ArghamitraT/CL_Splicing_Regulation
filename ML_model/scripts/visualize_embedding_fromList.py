from pathlib import Path
from omegaconf import OmegaConf
import os
import pickle
import numpy as np

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

def get_tsne_embedding(z0, z1):

    tsne = TSNE(n_components=2, perplexity=30)
    z0_2d = tsne.fit_transform(z0.cpu().numpy())
    z1_2d = tsne.fit_transform(z1.cpu().numpy())

    return z0_2d, z1_2d


def get_2view_embedding(config, device, view0, view1):
    model = load_pretrained_model(config, device)
    with torch.no_grad():
        if config.embedder._name_ == "MTSplice":
            z0 = model(view0[0].to(device), view0[1].to(device))
            z1 = model(view1[0].to(device), view1[1].to(device))
        else:
            z0 = model(view0.to(device))
            z1 = model(view1.to(device))

    # embeddings = torch.cat([z0, z1], dim=0).cpu().numpy()
    # tsne = TSNE(n_components=2, perplexity=30)
    # emb_2d = tsne.fit_transform(embeddings)

    # z0_2d = emb_2d[:z0.shape[0]]
    # z1_2d = emb_2d[z0.shape[0]:]

    # tsne = TSNE(n_components=2, perplexity=30)
    # z0_2d = tsne.fit_transform(z0.cpu().numpy())
    # z1_2d = tsne.fit_transform(z1.cpu().numpy())
    z0_2d, z1_2d = get_tsne_embedding(z0, z1)

    plt.figure(figsize=(8, 8))
    for i in range(z0.shape[0]):
        plt.plot([z0_2d[i, 0], z1_2d[i, 0]], [z0_2d[i, 1], z1_2d[i, 1]], color="gray", linewidth=0.5, alpha=0.5)
    plt.scatter(z0_2d[:, 0], z0_2d[:, 1], color="blue", s=20, label="Anchor (z0)")
    plt.scatter(z1_2d[:, 0], z1_2d[:, 1], color="orange", s=20, label="Positive (z1)")
    plt.legend()
    plt.title("t-SNE: Anchor–Positive Pairs")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f'{main_dir}/figures/tsne{time.strftime("_%Y_%m_%d__%H_%M_%S")}.png')

    # plt.savefig(f'../figures/tsne{time.strftime("_%Y_%m_%d__%H_%M_%S")}.png')




def all_pos_of_anchor(config, device, view0, train_loader, batch):
    
    model = load_pretrained_model(config, device)
    # anchor_idx = random.randint(0, len(view0) - 1)
    anchor_idx = 10
    # dataset = train_loader.dataset.dataset
    dataset = train_loader.dataset
    exon_name = dataset.exon_names[anchor_idx]
    all_views_dict = dataset.data[exon_name]

    augmentations = list(all_views_dict.values())
    

    if callable(tokenizer) and not hasattr(tokenizer, "vocab_size"):
        aug_tensor = torch.stack([
            tokenizer([seq])[0] for seq in augmentations
        ]).to(device)

    elif callable(tokenizer):  # HuggingFace-style
            aug_tensor = torch.stack([
            torch.tensor(tokenizer(seq)["input_ids"]) for seq in augmentations
        ]).to(device)
    else:
        print()

    other_indices = [i for i in range(len(view0)) if i != anchor_idx]
    view0_others = view0[other_indices].to(device)

    with torch.no_grad():
        z_anchor_aug = model(aug_tensor)
        z_others = model(view0_others)

    embeddings = torch.cat([z_anchor_aug, z_others], dim=0).cpu().numpy()
    labels = [0] * z_anchor_aug.shape[0] + [1] * z_others.shape[0]

    emb_2d = TSNE(n_components=2, perplexity=20).fit_transform(embeddings)

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=emb_2d[:, 0], y=emb_2d[:, 1], hue=labels, palette=["red", "gray"], s=30)
    plt.title(f"t-SNE: All Augmentations of One Anchor vs. Others\nid{anchor_idx}__{exon_name}")
    plt.axis("off")
    plt.tight_layout()
    plt.title(f'id{anchor_idx}__{exon_name}')
    # plt.savefig(f'{main_dir}figures/all_pos_of_anchor{time.strftime("_%Y_%m_%d__%H_%M_%S")}.png')
    plt.savefig(f'{main_dir}/figures/all_pos_of_anchor{time.strftime("_%Y_%m_%d__%H_%M_%S")}.png')


def distance_to_pos_and_neg(config, device, view0, train_loader, tokenizer):
    model = load_pretrained_model(config, device)
    anchor_idx = random.randint(0, len(view0) - 1)
    dataset = train_loader.dataset.dataset
    exon_name = dataset.exon_names[anchor_idx]
    all_views_dict = dataset.data[exon_name]

    augmentations = list(all_views_dict.values())
    aug_tensor = torch.stack([
        torch.tensor(tokenizer(seq)["input_ids"]) for seq in augmentations
    ]).to(device)

    other_indices = [i for i in range(len(view0)) if i != anchor_idx]
    view0_others = view0[other_indices].to(device)

    with torch.no_grad():
        z_anchor_aug = model(aug_tensor)
        anchor_vec = z_anchor_aug.mean(dim=0)
        z_others = model(view0_others)

    dist_to_pos = F.pairwise_distance(anchor_vec.unsqueeze(0), z_anchor_aug)
    dist_to_neg = F.pairwise_distance(anchor_vec.unsqueeze(0), z_others)

    print(f"Anchor: {exon_name} (idx {anchor_idx})")
    print(f"Distances to positives (mean): {dist_to_pos.mean():.4f}")
    print(f"Distances to negatives (mean): {dist_to_neg.mean():.4f}")


@torch.no_grad()
def plot_all_anchors_from_list(config, high_expr_exons, low_expr_exons, data_module, model, device, save_csv=True):
    """
    Follows EXACT same flow as training (Dataset -> Collate -> Tokenizer -> Encoder)
    and plots all exons + homologs together in ONE t-SNE figure,
    grouping exons into High vs Low expression sets.
    """

    import os
    import time
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import TSNE
    from src.datasets.utility import get_windows_with_padding
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

    normalize = True  # Whether to normalize embeddings before plotting

    # --- Setup directories ---
    timestamp = time.strftime("%Y_%m_%d__%H_%M_%S")
    out_dir = os.path.join(main_dir, f"figures/tsne_2exon_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)


    # --- Define high and low expression exons ---
    exon_list = high_expr_exons + low_expr_exons

    dataset = data_module.train_dataloader().dataset
    tokenizer = data_module.tokenizer
    os.makedirs(main_dir, exist_ok=True)

    all_embeddings = []
    all_species = []
    all_exons = []
    all_groups = []  # NEW: expression-level labels

    for exon_name in exon_list:
        if exon_name not in dataset.data:
            print(f"⚠️ Skipping {exon_name}: not found in dataset.")
            continue

        expr_group = "High" if exon_name in high_expr_exons else "Low"

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
            reduce_and_plot(name, coords, {"High": "#1f77b4", "Low": "#ff7f0e"})

        print(f"📊 All UMAP variants saved under: {out_dir}")

    # --- Save metrics metadata ---
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"📊 Saved metrics to {out_dir}/metrics.json")




# Define main directory for code files (e.g., for saving plots)
main_dir = str(CONTRASTIVE_ROOT / "code" / "ML_model")


# @hydra.main(version_base=None, config_path=get_config_path(), config_name="config.yaml")
@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def main(config: OmegaConf):
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
    
    # Example exon list (replace with your own)
  
    # high_expr_exons = [
    #     'ENST00000296411.11_8_11',
    #     'ENST00000224337.10_6_17',
    #     'ENST00000322507.13_43_65'
    # ]
    # low_expr_exons = [
    #     'ENST00000391859.6_5_5',
    #     'ENST00000367986.8_3_4',
    #     'ENST00000507462.5_3_3'
    # ]

    high_expr_exons = [
        'ENST00000296411.11_8_11',
        'ENST00000224337.10_6_17',
    ]
    low_expr_exons = [
        'ENST00000391859.6_5_5',
        'ENST00000507462.5_3_3'
    ]

    # --- Load the .pkl dict ---
    import pickle
    import random

    path = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/extra/filtered_HiLow_exons_by_tissue.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)

    plot_all_anchors_from_list(config, high_expr_exons, low_expr_exons, data_module, model, device)


    # # --- Aggregate all high/low expression exons across tissues ---
    # high_expr_exons, low_expr_exons = [], []

    # for tissue, values in data.items():
    #     high_list = values.get("high_expression_exons", [])
    #     low_list = values.get("low_expression_exons", [])
    #     if not high_list and not low_list:
    #         continue  # skip empty tissues

    #     print(f"{tissue:<40} | High: {len(high_list):3d} | Low: {len(low_list):3d}")
    #     high_expr_exons.extend(high_list)
    #     low_expr_exons.extend(low_list)

    # print(f"\n✅ Total high-expression exons: {len(high_expr_exons)}")
    # print(f"✅ Total low-expression exons:  {len(low_expr_exons)}")

    # # --- Randomly select 10 from each group ---
    # n_high = min(1, len(high_expr_exons))
    # n_low = min(1, len(low_expr_exons))

    # random_high = random.sample(high_expr_exons, n_high)
    # random_low = random.sample(low_expr_exons, n_low)

    # print(f"\n🎯 Selected {n_high} random high-expression exons:")
    # print(random_high)
    # print(f"\n🎯 Selected {n_low} random low-expression exons:")
    # print(random_low)

    # --- Plot the selected exons ---
    # plot_all_anchors_from_list(config, random_high, random_low, data_module, model, device)

    # --- Plot all high/low expression exons ---
    
if __name__ == "__main__":
    main()