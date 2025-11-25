"""
Visualization utilities for CLADES models.

Provides functionality for:
- t-SNE embeddings visualization
- UMAP embeddings visualization
- Prediction vs ground truth scatter plots
- Correlation heatmaps by tissue
- Training metrics tracking
- Embedding distribution analysis
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False
    warnings.warn("scikit-learn not installed. t-SNE visualization unavailable.")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("umap-learn not installed. UMAP visualization unavailable.")

from scipy.stats import spearmanr, pearsonr
import pytorch_lightning as pl


class CLADESVisualizer:
    """
    Comprehensive visualization toolkit for CLADES models.
    
    Supports:
    - Embedding space visualization (t-SNE, UMAP)
    - Prediction analysis (scatter plots, residuals)
    - Tissue-specific metrics
    - Training dynamics
    """
    
    def __init__(self, output_dir: str = "visualizations", figsize: Tuple[int, int] = (12, 9)):
        """
        Initialize visualizer.
        
        Parameters
        ----------
        output_dir : str
            Directory to save visualizations
        figsize : Tuple[int, int]
            Default figure size (width, height)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize

        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = figsize
    
    def visualize_tsne(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        perplexity: int = 30,
        n_iter: int = 1000,
        random_state: int = 42,
        title: str = "t-SNE Embedding Visualization",
        filename: Optional[str] = None,
        cmap: str = "viridis"
    ) -> Optional[np.ndarray]:
        """
        Visualize embeddings using t-SNE.
        
        Parameters
        ----------
        embeddings : np.ndarray
            High-dimensional embeddings (N, D)
        labels : Optional[np.ndarray]
            Labels for coloring points (N,). If None, no coloring applied.
        perplexity : int
            t-SNE perplexity parameter
        n_iter : int
            Number of t-SNE iterations
        random_state : int
            Random seed for reproducibility
        title : str
            Plot title
        filename : Optional[str]
            Save figure to this filename in output_dir
        cmap : str
            Colormap name
            
        Returns
        -------
        Optional[np.ndarray]
            t-SNE coordinates (N, 2), or None if sklearn unavailable
            
        Example
        -------
        >>> embeddings = model.get_embeddings(data)
        >>> viz.visualize_tsne(embeddings, labels=tissue_ids, 
        ...                     filename="tsne_embeddings.png")
        """
        if not TSNE_AVAILABLE:
            print("ERROR: scikit-learn required for t-SNE. Install with: pip install scikit-learn")
            return None
        
        print(f"Computing t-SNE for {embeddings.shape[0]} samples...")
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, embeddings.shape[0] - 1),
            n_iter=n_iter,
            random_state=random_state,
            verbose=1
        )
        tsne_results = tsne.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=self.figsize)
        
        if labels is not None:
            scatter = ax.scatter(
                tsne_results[:, 0],
                tsne_results[:, 1],
                c=labels,
                cmap=cmap,
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidth=0.5
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Label", fontsize=12)
        else:
            ax.scatter(
                tsne_results[:, 0],
                tsne_results[:, 1],
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidth=0.5
            )
        
        ax.set_xlabel("t-SNE 1", fontsize=12)
        ax.set_ylabel("t-SNE 2", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved t-SNE visualization to {filepath}")
        plt.close()
        
        return tsne_results
    
    def visualize_umap(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 42,
        title: str = "UMAP Embedding Visualization",
        filename: Optional[str] = None,
        cmap: str = "viridis"
    ) -> Optional[np.ndarray]:
        """
        Visualize embeddings using UMAP.
        
        Parameters
        ----------
        embeddings : np.ndarray
            High-dimensional embeddings (N, D)
        labels : Optional[np.ndarray]
            Labels for coloring points (N,)
        n_neighbors : int
            Number of neighbors for UMAP
        min_dist : float
            Minimum distance between points
        metric : str
            Distance metric
        random_state : int
            Random seed
        title : str
            Plot title
        filename : Optional[str]
            Save figure filename
        cmap : str
            Colormap name
            
        Returns
        -------
        Optional[np.ndarray]
            UMAP coordinates (N, 2), or None if umap unavailable
            
        Example
        -------
        >>> embeddings = model.get_embeddings(data)
        >>> viz.visualize_umap(embeddings, labels=psi_values,
        ...                    filename="umap_embeddings.png")
        """
        if not UMAP_AVAILABLE:
            print("ERROR: umap-learn required for UMAP. Install with: pip install umap-learn")
            return None
        
        print(f"Computing UMAP for {embeddings.shape[0]} samples...")
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            verbose=1
        )
        umap_results = reducer.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if labels is not None:
            scatter = ax.scatter(
                umap_results[:, 0],
                umap_results[:, 1],
                c=labels,
                cmap=cmap,
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidth=0.5
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Label", fontsize=12)
        else:
            ax.scatter(
                umap_results[:, 0],
                umap_results[:, 1],
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidth=0.5
            )
        
        ax.set_xlabel("UMAP 1", fontsize=12)
        ax.set_ylabel("UMAP 2", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved UMAP visualization to {filepath}")
        plt.close()
        
        return umap_results
    
    def plot_predictions_vs_ground_truth(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        title: str = "Predictions vs Ground Truth",
        filename: Optional[str] = None,
        tissues: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Scatter plot of predictions vs ground truth with metrics.
        
        Parameters
        ----------
        predictions : np.ndarray
            Model predictions (N,) or (N, T) for multiple tissues
        ground_truth : np.ndarray
            Ground truth values (N,) or (N, T)
        title : str
            Plot title
        filename : Optional[str]
            Save figure filename
        tissues : Optional[np.ndarray]
            Tissue identifiers for coloring
            
        Returns
        -------
        Dict[str, float]
            Dictionary with correlation and error metrics
            
        Example
        -------
        >>> preds = model.predict(test_data)
        >>> metrics = viz.plot_predictions_vs_ground_truth(
        ...     preds, test_labels, filename="pred_vs_truth.png"
        ... )
        >>> print(f"Spearman r: {metrics['spearman']:.3f}")
        """
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()
        if len(ground_truth.shape) > 1:
            ground_truth = ground_truth.flatten()

        valid_idx = ~(np.isnan(predictions) | np.isnan(ground_truth))
        predictions = predictions[valid_idx]
        ground_truth = ground_truth[valid_idx]
        tissues = tissues[valid_idx] if tissues is not None else None

        spearman_r, spearman_p = spearmanr(predictions, ground_truth)
        pearson_r, pearson_p = pearsonr(predictions, ground_truth)
        mse = np.mean((predictions - ground_truth) ** 2)
        mae = np.mean(np.abs(predictions - ground_truth))
        
        metrics = {
            'spearman': spearman_r,
            'spearman_pval': spearman_p,
            'pearson': pearson_r,
            'pearson_pval': pearson_p,
            'mse': mse,
            'mae': mae
        }

        fig, ax = plt.subplots(figsize=self.figsize)

        if tissues is not None:
            unique_tissues = np.unique(tissues)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_tissues)))
            for tissue, color in zip(unique_tissues, colors):
                mask = tissues == tissue
                ax.scatter(ground_truth[mask], predictions[mask],
                          label=tissue, alpha=0.6, s=50, color=color)
            ax.legend(title="Tissue", bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.scatter(ground_truth, predictions, alpha=0.6, s=50, color='steelblue')

        lims = [
            np.min([ground_truth.min(), predictions.min()]),
            np.max([ground_truth.max(), predictions.max()])
        ]
        ax.plot(lims, lims, 'k--', alpha=0.3, lw=2, label='Perfect Prediction')

        ax.set_xlabel("Ground Truth", fontsize=12)
        ax.set_ylabel("Predictions", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        textstr = f"Spearman r: {spearman_r:.3f}\nPearson r: {pearson_r:.3f}\nMAE: {mae:.4f}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved prediction plot to {filepath}")
        plt.close()
        
        return metrics
    
    def plot_residuals(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        title: str = "Prediction Residuals",
        filename: Optional[str] = None
    ) -> None:
        """
        Analyze and plot residuals (prediction errors).
        
        Parameters
        ----------
        predictions : np.ndarray
            Model predictions
        ground_truth : np.ndarray
            Ground truth values
        title : str
            Plot title
        filename : Optional[str]
            Save figure filename
            
        Example
        -------
        >>> viz.plot_residuals(preds, ground_truth, 
        ...                    filename="residuals.png")
        """
        predictions = predictions.flatten()
        ground_truth = ground_truth.flatten()
        valid_idx = ~(np.isnan(predictions) | np.isnan(ground_truth))
        predictions = predictions[valid_idx]
        ground_truth = ground_truth[valid_idx]
        
        residuals = predictions - ground_truth
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)

        axes[0, 0].scatter(ground_truth, residuals, alpha=0.6, s=30)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel("Ground Truth")
        axes[0, 0].set_ylabel("Residuals")
        axes[0, 0].set_title("Residuals vs Ground Truth")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].scatter(predictions, residuals, alpha=0.6, s=30)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel("Predictions")
        axes[0, 1].set_ylabel("Residuals")
        axes[0, 1].set_title("Residuals vs Predictions")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel("Residual Value")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Distribution of Residuals")
        
        # Q-Q plot (residuals)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title("Q-Q Plot")
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved residual plot to {filepath}")
        plt.close()
    
    def plot_tissue_metrics_heatmap(
        self,
        tissue_metrics: Dict[str, Dict[str, float]],
        title: str = "Metrics by Tissue",
        filename: Optional[str] = None
    ) -> None:
        """
        Heatmap of metrics across tissues.
        
        Parameters
        ----------
        tissue_metrics : Dict[str, Dict[str, float]]
            Nested dict: tissue -> metric -> value
            Example: {"heart": {"spearman": 0.95, "mse": 0.01}, ...}
        title : str
            Plot title
        filename : Optional[str]
            Save figure filename
            
        Example
        -------
        >>> metrics = {"tissue1": {"spearman": 0.9, "mse": 0.01},
        ...            "tissue2": {"spearman": 0.85, "mse": 0.02}}
        >>> viz.plot_tissue_metrics_heatmap(metrics, 
        ...                                 filename="tissue_heatmap.png")
        """
        df = pd.DataFrame(tissue_metrics).T
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5,
                   cbar_kws={'label': 'Metric Value'}, ax=ax, linewidths=1)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Metrics", fontsize=12)
        ax.set_ylabel("Tissues", fontsize=12)
        
        plt.tight_layout()
        
        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved tissue metrics heatmap to {filepath}")
        plt.close()
    
    def plot_embedding_statistics(
        self,
        embeddings: np.ndarray,
        title: str = "Embedding Distribution Statistics",
        filename: Optional[str] = None
    ) -> None:
        """
        Analyze and visualize embedding space statistics.
        
        Parameters
        ----------
        embeddings : np.ndarray
            Embeddings (N, D)
        title : str
            Plot title
        filename : Optional[str]
            Save figure filename
            
        Example
        -------
        >>> embeddings = model.encoder(data)
        >>> viz.plot_embedding_statistics(embeddings.cpu().numpy(),
        ...                                filename="embedding_stats.png")
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # L2 norm distribution
        norms = np.linalg.norm(embeddings, axis=1)
        axes[0, 0].hist(norms, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel("L2 Norm")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title(f"L2 Norm Distribution (mean={norms.mean():.3f})")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mean absolute value
        mean_abs = np.mean(np.abs(embeddings), axis=0)
        axes[0, 1].plot(mean_abs, linewidth=1, marker='o', markersize=3)
        axes[0, 1].set_xlabel("Dimension")
        axes[0, 1].set_ylabel("Mean Absolute Value")
        axes[0, 1].set_title("Per-Dimension Mean Absolute Value")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Variance by dimension
        variances = np.var(embeddings, axis=0)
        axes[1, 0].plot(variances, linewidth=1, marker='o', markersize=3, color='orange')
        axes[1, 0].set_xlabel("Dimension")
        axes[1, 0].set_ylabel("Variance")
        axes[1, 0].set_title("Per-Dimension Variance")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Correlation matrix (for first 20 dims if high dimensional)
        if embeddings.shape[1] > 20:
            corr_embeddings = embeddings[:, :20]
        else:
            corr_embeddings = embeddings
        
        corr = np.corrcoef(corr_embeddings.T)
        im = axes[1, 1].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 1].set_title(f"Correlation Matrix ({corr_embeddings.shape[1]} dims)")
        plt.colorbar(im, ax=axes[1, 1], label='Correlation')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved embedding statistics to {filepath}")
        plt.close()
    
    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_metrics: Optional[Dict[str, List[float]]] = None,
        title: str = "Training Dynamics",
        filename: Optional[str] = None
    ) -> None:
        """
        Plot training and validation curves.
        
        Parameters
        ----------
        train_losses : List[float]
            Training loss at each epoch
        val_losses : List[float]
            Validation loss at each epoch
        train_metrics : Optional[Dict[str, List[float]]]
            Additional metrics to plot (e.g., {"accuracy": [...], "f1": [...]})
        title : str
            Plot title
        filename : Optional[str]
            Save figure filename
            
        Example
        -------
        >>> losses = trainer.state.loss_history
        >>> viz.plot_training_curves(losses['train'], losses['val'],
        ...                          filename="training_curves.png")
        """
        num_plots = 1 + (len(train_metrics) if train_metrics else 0)
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 4))
        
        if num_plots == 1:
            axes = [axes]
        
        epochs = np.arange(1, len(train_losses) + 1)

        axes[0].plot(epochs, train_losses, 'o-', label='Train Loss', linewidth=2, markersize=4)
        axes[0].plot(epochs, val_losses, 's-', label='Val Loss', linewidth=2, markersize=4)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        if train_metrics:
            for idx, (metric_name, metric_values) in enumerate(train_metrics.items(), 1):
                axes[idx].plot(epochs, metric_values, 'o-', linewidth=2, markersize=4)
                axes[idx].set_xlabel("Epoch")
                axes[idx].set_ylabel(metric_name.replace('_', ' ').title())
                axes[idx].set_title(f"{metric_name.replace('_', ' ').title()} Over Training")
                axes[idx].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved training curves to {filepath}")
        plt.close()
    
    def create_checkpoint_summary_report(
        self,
        checkpoint_path: str,
        model: Optional[pl.LightningModule] = None,
        predictions: Optional[np.ndarray] = None,
        ground_truth: Optional[np.ndarray] = None,
        embeddings: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        tissue_metrics: Optional[Dict] = None,
        filename_prefix: str = "checkpoint_report"
    ) -> None:
        """
        Create a comprehensive report for a checkpoint.
        
        Parameters
        ----------
        checkpoint_path : str
            Path to model checkpoint
        model : Optional[pl.LightningModule]
            Model instance (for getting embeddings)
        predictions : Optional[np.ndarray]
            Model predictions
        ground_truth : Optional[np.ndarray]
            Ground truth values
        embeddings : Optional[np.ndarray]
            Pre-computed embeddings
        labels : Optional[np.ndarray]
            Labels for embeddings
        tissue_metrics : Optional[Dict]
            Tissue-specific metrics
        filename_prefix : str
            Prefix for generated files
            
        Example
        -------
        >>> viz.create_checkpoint_summary_report(
        ...     checkpoint_path="checkpoints/best.ckpt",
        ...     predictions=preds,
        ...     ground_truth=truth,
        ...     embeddings=embeds,
        ...     filename_prefix="best_model"
        ... )
        """
        print(f"\n{'='*60}")
        print(f"Creating summary report for: {checkpoint_path}")
        print(f"{'='*60}\n")
        
        if predictions is None:
            checkpoint_path_obj = Path(checkpoint_path)
            checkpoints_dir = checkpoint_path_obj.parent.parent.parent.parent
            print(f"Searching for training outputs in: {checkpoints_dir}\n")
            
            for pattern in ['*_predictions*.tsv', '*_outputs*.tsv', '*_raw_output*.tsv']:
                matching_files = sorted(list(checkpoints_dir.glob(pattern)))
                if matching_files:
                    try:
                        pred_file = matching_files[-1]
                        print(f"Loading predictions from: {pred_file}")
                        pred_df = pd.read_csv(pred_file, sep='\t')
                        predictions = pred_df.iloc[:, 1:].values.astype(float)
                        print(f"Loaded {predictions.shape[0]} samples\n")
                        break
                    except Exception as e:
                        print(f"Could not load: {e}\n")

            if tissue_metrics is None:
                for pattern in ['*_spearman*.tsv', '*_metrics*.tsv']:
                    matching_files = sorted(list(checkpoints_dir.glob(pattern)))
                    if matching_files:
                        try:
                            metrics_file = matching_files[-1]
                            print(f"Loading metrics from: {metrics_file}")
                            metrics_df = pd.read_csv(metrics_file, sep='\t', index_col=0)
                            tissue_metrics = {str(tissue): dict(metrics_df.loc[tissue]) for tissue in metrics_df.index}
                            print(f"Loaded metrics for {len(tissue_metrics)} tissues\n")
                            break
                        except Exception as e:
                            print(f"Could not load: {e}\n")

        if embeddings is None:
            try:
                checkpoint_path_obj = Path(checkpoint_path)
                checkpoint_dir = checkpoint_path_obj.parent.parent.parent
                
                embeddings_file = checkpoint_dir / "embeddings.npy"
                if embeddings_file.exists():
                    print(f"Loading pre-computed embeddings from: {embeddings_file}")
                    embeddings = np.load(embeddings_file)
                    print(f"Loaded embeddings shape: {embeddings.shape}\n")
                else:
                    print("Note: Pre-computed embeddings not found.")
                    print(f"  Expected at: {embeddings_file}")
                    print("  To extract embeddings, run:")
                    print(f"    python scripts/extract_embeddings.py \\")
                    print(f"      --checkpoint {checkpoint_path} \\")
                    print(f"      --data <csv_with_sequences> \\")
                    print(f"      --output {embeddings_file}\n")
            except Exception as e:
                print(f"Could not load embeddings: {e}\n")

        if predictions is not None and ground_truth is not None:
            print("[1/5] Generating prediction visualizations...")
            self.plot_predictions_vs_ground_truth(
                predictions, ground_truth,
                filename=f"{filename_prefix}_pred_vs_truth.png"
            )
            self.plot_residuals(
                predictions, ground_truth,
                filename=f"{filename_prefix}_residuals.png"
            )
        elif predictions is not None:
            print("[1/5] Generating prediction statistics...")
            self.plot_embedding_statistics(
                predictions,
                filename=f"{filename_prefix}_prediction_stats.png"
            )
        else:
            print("[1/5] Skipping predictions (no data)")

        if embeddings is not None:
            print("[2/5] Generating UMAP...")
            self.visualize_umap(
                embeddings, labels=labels,
                filename=f"{filename_prefix}_umap.png"
            )
            
            print("[3/5] Generating t-SNE...")
            if TSNE_AVAILABLE:
                self.visualize_tsne(
                    embeddings, labels=labels,
                    filename=f"{filename_prefix}_tsne.png"
                )
            
            print("[4/5] Generating embedding stats...")
            self.plot_embedding_statistics(
                embeddings,
                filename=f"{filename_prefix}_embedding_stats.png"
            )
        else:
            print("[2/5] Skipping embeddings (no data)")
        
        # Tissue metrics
        if tissue_metrics is not None:
            print("[5/5] Generating tissue metrics heatmap...")
            self.plot_tissue_metrics_heatmap(
                tissue_metrics,
                filename=f"{filename_prefix}_tissue_metrics.png"
            )
        else:
            print("[5/5] Skipping metrics (no data)")
        
        print(f"\n{'='*60}")
        print(f"Report complete! Saved to: {self.output_dir}")
        print(f"{'='*60}\n")


def extract_embeddings_from_checkpoint(
    checkpoint_path: str,
    dataloader,
    model_class,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract embeddings from a checkpoint.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to model checkpoint
    dataloader : DataLoader
        DataLoader with data
    model_class : type
        Model class to instantiate
    device : str
        Device to run model on
        
    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray]]
        Embeddings (N, D) and labels if available
    """
    model = model_class.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()
    
    embeddings_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
                y = batch[1] if len(batch) > 1 else None
            else:
                x = batch.to(device)
                y = None

            embed = model.encoder(x)
            embeddings_list.append(embed.cpu().numpy())
            
            if y is not None:
                if isinstance(y, torch.Tensor):
                    labels_list.append(y.cpu().numpy())
    
    embeddings = np.vstack(embeddings_list)
    labels = np.hstack(labels_list) if labels_list else None
    
    return embeddings, labels


if __name__ == "__main__":
    """
    Command-line interface for generating visualizations from checkpoints.
    
    Usage:
        python visualizer.py --checkpoint <path> --output <dir>
        python visualizer.py --help
    """
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Generate visualizations from CLADES model checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualizer.py --checkpoint output/finetune_2025_11_14_23_46_21/checkpoints/psi_regression/MTSplice/400/best-checkpoint.ckpt --output output/visualizations
  python visualizer.py --checkpoint checkpoints/best.ckpt
        """
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/visualizations",
        help="Output directory for visualizations (default: output/visualizations)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="checkpoint_report",
        help="Prefix for generated filenames (default: checkpoint_report)"
    )
    
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    if not str(checkpoint_path).endswith('.ckpt'):
        print("ERROR: Checkpoint must be a .ckpt file")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("CLADES Visualizer - Checkpoint Report Generator")
    print(f"{'='*60}\n")
    
    viz = CLADESVisualizer(output_dir=args.output)

    viz.create_checkpoint_summary_report(
        checkpoint_path=str(checkpoint_path),
        filename_prefix=args.prefix
    )

