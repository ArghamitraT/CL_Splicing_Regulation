#!/usr/bin/env python3
"""
Visualize extracted embeddings using t-SNE and UMAP.

Uses embeddings saved from extract_embeddings_dilated_conv.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualizer import CLADESVisualizer


def load_embeddings(embedding_dir: str, split: str = "train"):
    """Load saved embeddings and metadata."""
    embedding_dir = Path(embedding_dir)
    
    embeddings = np.load(embedding_dir / f"embeddings_{split}.npy")
    exon_ids = np.load(embedding_dir / f"exon_ids_{split}.npy")
    exon_names = np.load(embedding_dir / f"exon_names_{split}.npy")
    
    print(f"Loaded {len(embeddings)} embeddings from {embedding_dir}")
    print(f"Embedding shape: {embeddings.shape}")
    
    return embeddings, exon_ids, exon_names


def visualize_embeddings(
    embedding_dir: str,
    output_dir: str,
    split: str = "train",
    perplexity: int = 30,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    random_state: int = 42,
):
    """
    Visualize embeddings using t-SNE and UMAP.
    
    Args:
        embedding_dir: Directory containing saved embeddings
        output_dir: Directory to save visualizations
        split: Which split to visualize ('train', 'val', 'test')
        perplexity: t-SNE perplexity parameter
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        n_components: Number of dimensions for reduction (default 2)
        random_state: Random seed for reproducibility
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    embeddings, exon_ids, exon_names = load_embeddings(embedding_dir, split)
    
    print(f"\nVisualizing {len(embeddings)} embeddings...")
    
    # t-SNE visualization
    print("\nRunning t-SNE...")
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=1000,
        verbose=1
    )
    tsne_embeddings = tsne.fit_transform(embeddings)
    
    print(f"t-SNE shape: {tsne_embeddings.shape}")
    
    # UMAP visualization
    print("\nRunning UMAP...")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        verbose=True
    )
    umap_embeddings = reducer.fit_transform(embeddings)
    
    print(f"UMAP shape: {umap_embeddings.shape}")
    
    print("\nGenerating plots...")
    
    # t-SNE plot
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    scatter = ax.scatter(
        tsne_embeddings[:, 0],
        tsne_embeddings[:, 1],
        c=range(len(embeddings)),
        cmap='tab20',
        s=30,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.3
    )
    ax.set_xlabel('t-SNE 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE 2', fontsize=12, fontweight='bold')
    ax.set_title(f't-SNE Visualization of DilatedConv1D Embeddings ({split} set, n={len(embeddings)})', 
                 fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Sample Index')
    plt.tight_layout()
    
    tsne_path = output_dir / f"tsne_{split}.png"
    plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
    print(f"Saved t-SNE plot to {tsne_path}")
    plt.close()
    
    # UMAP plot
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    scatter = ax.scatter(
        umap_embeddings[:, 0],
        umap_embeddings[:, 1],
        c=range(len(embeddings)),
        cmap='tab20',
        s=30,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.3
    )
    ax.set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
    ax.set_title(f'UMAP Visualization of DilatedConv1D Embeddings ({split} set, n={len(embeddings)})', 
                 fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Sample Index')
    plt.tight_layout()
    
    umap_path = output_dir / f"umap_{split}.png"
    plt.savefig(umap_path, dpi=300, bbox_inches='tight')
    print(f"Saved UMAP plot to {umap_path}")
    plt.close()
    
    np.save(output_dir / f"tsne_embeddings_{split}.npy", tsne_embeddings)
    np.save(output_dir / f"umap_embeddings_{split}.npy", umap_embeddings)
    
    print(f"\nVisualization complete! Plots saved to {output_dir}")
    print(f"  - t-SNE: {tsne_path}")
    print(f"  - UMAP: {umap_path}")
    print(f"  - Reduced embeddings also saved as .npy files")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize extracted embeddings using t-SNE and UMAP"
    )
    parser.add_argument(
        "--embedding-dir",
        type=str,
        default="output/embeddings_dilated",
        help="Directory containing extracted embeddings"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/visualizations_dilated",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Which split to visualize"
    )
    parser.add_argument(
        "--perplexity",
        type=int,
        default=30,
        help="t-SNE perplexity parameter"
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter"
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter"
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        help="Number of dimensions for reduction"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    visualize_embeddings(
        embedding_dir=args.embedding_dir,
        output_dir=args.output_dir,
        split=args.split,
        perplexity=args.perplexity,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=args.n_components,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
