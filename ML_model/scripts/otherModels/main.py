
from visualize.tsne_plot import plot_2view_tsne
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedder_name = "borzoi"  # or "enformer"

if embedder_name == "borzoi":
    from borzoi_pytorch import Borzoi
    embedder = Borzoi.from_pretrained('johahi/borzoi-replicate-0')
    print(embedder.config)
    embedder.to(device)
    embedder.eval()

    # 2 dummy sequences
    seqs = ["A" * 200]
    view0 = torch.randn(1, 4, 200).to(device)    # Working lengths: greater than or equal to 196,608
    with torch.no_grad():
        embeddings = embedder(view0)

    print(embeddings.shape)

    # plot_2view_tsne_numpy(z0, z1, save_path="tsne_borzoi.png")

elif embedder_name == "enformer":
    from embedder.enformer_embedder import EnformerEmbedder
    encoder = EnformerEmbedder(device)

    # Fake 1-hot PyTorch input
    view0 = torch.randn(16, 1000, 4)
    view1 = torch.randn(16, 1000, 4)

    plot_2view_tsne(encoder, view0, view1, save_path="tsne_enformer.png")

else:
    raise ValueError("Unknown embedder")
