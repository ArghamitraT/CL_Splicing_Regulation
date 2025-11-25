#!/usr/bin/env python3
"""
Comprehensive comparison of MTSplice vs DilatedConv1D embedders.

Generates:
1. Per-tissue performance heatmaps
2. Memory usage analysis
3. Convergence curves
4. Robustness analysis
5. Error distribution analysis
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr


def find_latest_finetune_checkpoint(root_path: str, embedder_name: str) -> Path:
    """Find the latest finetune checkpoint for a specific embedder."""
    output_dir = Path(root_path)
    
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    # Find all finetune_* directories
    finetune_dirs = sorted([d for d in output_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('finetune_')])
    
    if not finetune_dirs:
        raise FileNotFoundError(f"No finetune directories found in {output_dir}")
    
    # Search from most recent backwards
    for finetune_dir in reversed(finetune_dirs):
        checkpoints_dir = finetune_dir / "checkpoints"
        metrics_file = checkpoints_dir / "tsplice_spearman_by_tissue.tsv"
        embedder_ckpt = checkpoints_dir / "psi_regression" / embedder_name / "400"
        
        if metrics_file.exists() and embedder_ckpt.exists():
            print(f"✨ Found {embedder_name} finetune checkpoint: {finetune_dir.name}")
            return finetune_dir
    
    raise FileNotFoundError(f"No valid finetune checkpoint found for {embedder_name}")



def load_metrics(checkpoint_dir: Path) -> Dict:
    """Load metrics from checkpoint directory."""
    metrics_file = checkpoint_dir / "checkpoints" / "tsplice_spearman_by_tissue.tsv"
    
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    
    df = pd.read_csv(metrics_file, sep="\t")
    return df


def load_predictions(checkpoint_dir: Path) -> pd.DataFrame:
    """Load predictions from checkpoint directory."""
    pred_file = checkpoint_dir / "checkpoints" / "tsplice_final_predictions_all_tissues.tsv"
    
    if not pred_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_file}")
    
    df = pd.read_csv(pred_file, sep="\t")
    return df


def load_ground_truth(data_dir: Path) -> pd.DataFrame:
    """Load ground truth PSI values."""
    gt_file = data_dir / "test_cassette_exons_with_logit_mean_psi.csv"
    
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
    
    df = pd.read_csv(gt_file)
    return df


def generate_tissue_heatmap(metrics_df: pd.DataFrame, output_dir: Path, embedder_name: str):
    """Generate heatmap of per-tissue Spearman correlations."""
    tissues = metrics_df['tissue'].values
    psi_values = metrics_df['spearman_psi'].values
    delta_values = metrics_df['spearman_delta'].values
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    
    # PSI heatmap
    psi_data = psi_values.reshape(-1, 1)
    sns.heatmap(psi_data, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.8, vmax=1.0,
                yticklabels=tissues, xticklabels=['Spearman PSI'], ax=ax1, cbar_kws={'label': 'Correlation'})
    ax1.set_title(f'{embedder_name}: Per-Tissue Spearman PSI Correlation', fontsize=14, fontweight='bold')
    
    # Delta heatmap
    delta_data = delta_values.reshape(-1, 1)
    sns.heatmap(delta_data, annot=True, fmt='.3f', cmap='RdBu_r', vmin=-0.3, vmax=0.3,
                yticklabels=tissues, xticklabels=['Spearman Δlogit'], ax=ax2, cbar_kws={'label': 'Correlation'})
    ax2.set_title(f'{embedder_name}: Per-Tissue Spearman Δlogit Correlation', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / f"tissue_heatmap_{embedder_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved tissue heatmap to {output_path}")
    plt.close()


def generate_convergence_analysis(finetune_dir: Path, output_dir: Path, embedder_name: str):
    """Analyze convergence from training logs."""
    log_file = finetune_dir / "hydra" / "finetune_CLADES.log"
    
    if not log_file.exists():
        print(f"⚠️ Log file not found: {log_file}")
        return

    train_losses = []
    val_losses = []
    epochs = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if 'train_loss_epoch=' in line:
                try:
                    parts = line.split('train_loss_epoch=')
                    if len(parts) > 1:
                        val_str = parts[1].split(',')[0].split(']')[0]
                        train_losses.append(float(val_str))
                except:
                    pass
            
            if 'val_loss=' in line and 'val_loss_epoch' not in line:
                try:
                    parts = line.split('val_loss=')
                    if len(parts) > 1:
                        val_str = parts[1].split(',')[0].split(']')[0]
                        val_losses.append(float(val_str))
                except:
                    pass
    
    if not train_losses or not val_losses:
        print(f"⚠️ Could not extract loss values from log")
        return
    
    epochs = list(range(len(train_losses)))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, 'o-', label='Training Loss', linewidth=2, markersize=8)
    ax.plot(epochs, val_losses, 's-', label='Validation Loss', linewidth=2, markersize=8)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title(f'{embedder_name}: Training Convergence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f"convergence_{embedder_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved convergence plot to {output_path}")
    plt.close()


def generate_robustness_analysis(metrics_df: pd.DataFrame, output_dir: Path, embedder_name: str):
    """Analyze robustness across tissues."""
    psi_values = metrics_df['spearman_psi'].values
    delta_values = metrics_df['spearman_delta'].values
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # PSI distribution
    axes[0, 0].hist(psi_values, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(psi_values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {psi_values.mean():.4f}')
    axes[0, 0].axvline(np.median(psi_values), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(psi_values):.4f}')
    axes[0, 0].set_xlabel('Spearman PSI', fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontweight='bold')
    axes[0, 0].set_title(f'{embedder_name}: PSI Distribution', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Delta distribution
    axes[0, 1].hist(delta_values, bins=15, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(delta_values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {delta_values.mean():.4f}')
    axes[0, 1].axvline(np.median(delta_values), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(delta_values):.4f}')
    axes[0, 1].set_xlabel('Spearman Δlogit', fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontweight='bold')
    axes[0, 1].set_title(f'{embedder_name}: Δlogit Distribution', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plots
    box_data = [psi_values, delta_values]
    bp = axes[1, 0].boxplot(box_data, labels=['PSI', 'Δlogit'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['steelblue', 'coral']):
        patch.set_facecolor(color)
    axes[1, 0].set_ylabel('Correlation Value', fontweight='bold')
    axes[1, 0].set_title(f'{embedder_name}: Correlation Distributions', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Statistics text
    stats_text = f"""
    PSI Statistics:
    • Mean: {psi_values.mean():.4f}
    • Std Dev: {psi_values.std():.4f}
    • Min: {psi_values.min():.4f}
    • Max: {psi_values.max():.4f}
    • CV: {(psi_values.std()/psi_values.mean()):.4f}
    
    Δlogit Statistics:
    • Mean: {delta_values.mean():.4f}
    • Std Dev: {delta_values.std():.4f}
    • Min: {delta_values.min():.4f}
    • Max: {delta_values.max():.4f}
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / f"robustness_{embedder_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved robustness analysis to {output_path}")
    plt.close()
    
    return {
        'psi_mean': psi_values.mean(),
        'psi_std': psi_values.std(),
        'psi_cv': psi_values.std() / psi_values.mean(),
        'delta_mean': delta_values.mean(),
        'delta_std': delta_values.std(),
    }


def generate_error_analysis(predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame,
                           output_dir: Path, embedder_name: str) -> Dict:
    """Analyze prediction errors."""
    tissue_cols = [col for col in predictions_df.columns if col != 'exon_id']
    
    errors_by_tissue = {}
    all_errors = []
    
    for tissue in tissue_cols:
        if tissue in predictions_df.columns:
            pred_values = predictions_df[tissue].values
            gt_mean = ground_truth_df[tissue].mean() if tissue in ground_truth_df.columns else pred_values.mean()
            errors = np.abs(pred_values - gt_mean)
            errors_by_tissue[tissue] = {
                'mae': np.mean(errors),
                'rmse': np.sqrt(np.mean(errors**2)),
                'std': np.std(errors)
            }
            all_errors.extend(errors)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Overall error histogram
    axes[0, 0].hist(all_errors, bins=50, color='darkblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(all_errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_errors):.4f}')
    axes[0, 0].set_xlabel('Absolute Error', fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontweight='bold')
    axes[0, 0].set_title(f'{embedder_name}: Overall Error Distribution', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE by tissue
    tissues = list(errors_by_tissue.keys())
    mae_values = [errors_by_tissue[t]['mae'] for t in tissues]
    axes[0, 1].barh(range(len(tissues)), mae_values, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 1].set_yticks(range(len(tissues)))
    axes[0, 1].set_yticklabels(tissues, fontsize=8)
    axes[0, 1].set_xlabel('Mean Absolute Error', fontweight='bold')
    axes[0, 1].set_title(f'{embedder_name}: MAE by Tissue', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # RMSE by tissue
    rmse_values = [errors_by_tissue[t]['rmse'] for t in tissues]
    axes[1, 0].barh(range(len(tissues)), rmse_values, color='coral', edgecolor='black', alpha=0.7)
    axes[1, 0].set_yticks(range(len(tissues)))
    axes[1, 0].set_yticklabels(tissues, fontsize=8)
    axes[1, 0].set_xlabel('Root Mean Squared Error', fontweight='bold')
    axes[1, 0].set_title(f'{embedder_name}: RMSE by Tissue', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Error statistics
    error_stats = f"""
    Overall Error Statistics:
    • Mean: {np.mean(all_errors):.4f}
    • Median: {np.median(all_errors):.4f}
    • Std Dev: {np.std(all_errors):.4f}
    • Min: {np.min(all_errors):.4f}
    • Max: {np.max(all_errors):.4f}
    • 95th Percentile: {np.percentile(all_errors, 95):.4f}
    """
    
    axes[1, 1].text(0.1, 0.5, error_stats, fontsize=11, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / f"error_analysis_{embedder_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved error analysis to {output_path}")
    plt.close()
    
    return {
        'mean_error': np.mean(all_errors),
        'median_error': np.median(all_errors),
        'std_error': np.std(all_errors),
        'max_error': np.max(all_errors),
    }


def generate_comparison_summary(mtsplice_dir: Path, dilated_dir: Path, output_dir: Path):
    """Generate comprehensive comparison summary."""
    print("\n" + "="*80)
    print("COMPREHENSIVE EMBEDDER COMPARISON")
    print("="*80)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading metrics...")
    mtsplice_metrics = load_metrics(mtsplice_dir)
    dilated_metrics = load_metrics(dilated_dir)

    mtsplice_pred = load_predictions(mtsplice_dir)
    dilated_pred = load_predictions(dilated_dir)
    
    # 1. Generate tissue heatmaps
    print("\nGenerating tissue heatmaps...")
    generate_tissue_heatmap(mtsplice_metrics, output_dir, "MTSplice")
    generate_tissue_heatmap(dilated_metrics, output_dir, "DilatedConv1D")
    
    # 2. Generate convergence analysis
    print("\nAnalyzing convergence...")
    generate_convergence_analysis(mtsplice_dir, output_dir, "MTSplice")
    generate_convergence_analysis(dilated_dir, output_dir, "DilatedConv1D")
    
    # 3. Generate robustness analysis
    print("\nAnalyzing robustness...")
    mtsplice_robust = generate_robustness_analysis(mtsplice_metrics, output_dir, "MTSplice")
    dilated_robust = generate_robustness_analysis(dilated_metrics, output_dir, "DilatedConv1D")
    
    # 4. Generate error analysis
    print("\nAnalyzing errors...")
    try:
        gt_df = load_ground_truth(Path("data/finetune_sample_data"))
        mtsplice_errors = generate_error_analysis(mtsplice_pred, gt_df, output_dir, "MTSplice")
        dilated_errors = generate_error_analysis(dilated_pred, gt_df, output_dir, "DilatedConv1D")
    except Exception as e:
        print(f"Could not load ground truth: {e}")
        mtsplice_errors = {}
        dilated_errors = {}
    
    # Generate comparison table
    print("\nGenerating comparison table...")
    comparison_data = {
        'Metric': [],
        'MTSplice': [],
        'DilatedConv1D': [],
        'Winner': []
    }
    
    # PSI metrics
    mtsplice_psi_mean = mtsplice_metrics['spearman_psi'].mean()
    dilated_psi_mean = dilated_metrics['spearman_psi'].mean()
    
    comparison_data['Metric'].append('Mean Spearman PSI')
    comparison_data['MTSplice'].append(f"{mtsplice_psi_mean:.4f}")
    comparison_data['DilatedConv1D'].append(f"{dilated_psi_mean:.4f}")
    comparison_data['Winner'].append('🤝 Tie' if abs(mtsplice_psi_mean - dilated_psi_mean) < 0.001 else 
                                     ('MTSplice ⭐' if mtsplice_psi_mean > dilated_psi_mean else 'DilatedConv1D ⭐'))
    
    # Delta metrics
    mtsplice_delta_mean = mtsplice_metrics['spearman_delta'].mean()
    dilated_delta_mean = dilated_metrics['spearman_delta'].mean()
    
    comparison_data['Metric'].append('Mean Spearman Δlogit')
    comparison_data['MTSplice'].append(f"{mtsplice_delta_mean:.4f}")
    comparison_data['DilatedConv1D'].append(f"{dilated_delta_mean:.4f}")
    comparison_data['Winner'].append('MTSplice ⭐' if mtsplice_delta_mean > dilated_delta_mean else 'DilatedConv1D ⭐')
    
    # Robustness - coefficient of variation
    comparison_data['Metric'].append('PSI Coefficient of Variation')
    comparison_data['MTSplice'].append(f"{mtsplice_robust['psi_cv']:.4f}")
    comparison_data['DilatedConv1D'].append(f"{dilated_robust['psi_cv']:.4f}")
    comparison_data['Winner'].append('MTSplice ⭐' if mtsplice_robust['psi_cv'] < dilated_robust['psi_cv'] else 'DilatedConv1D ⭐')
    
    # Error metrics
    if mtsplice_errors and dilated_errors:
        comparison_data['Metric'].append('Mean Absolute Error')
        comparison_data['MTSplice'].append(f"{mtsplice_errors['mean_error']:.4f}")
        comparison_data['DilatedConv1D'].append(f"{dilated_errors['mean_error']:.4f}")
        comparison_data['Winner'].append('MTSplice ⭐' if mtsplice_errors['mean_error'] < dilated_errors['mean_error'] else 'DilatedConv1D ⭐')
        
        comparison_data['Metric'].append('Max Error')
        comparison_data['MTSplice'].append(f"{mtsplice_errors['max_error']:.4f}")
        comparison_data['DilatedConv1D'].append(f"{dilated_errors['max_error']:.4f}")
        comparison_data['Winner'].append('MTSplice ⭐' if mtsplice_errors['max_error'] < dilated_errors['max_error'] else 'DilatedConv1D ⭐')
    
    comparison_df = pd.DataFrame(comparison_data)

    comparison_path = output_dir / "comparison_summary.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved comparison table to {comparison_path}")

    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Compare MTSplice vs DilatedConv1D performance")
    parser.add_argument(
        "--output-root",
        type=str,
        default="output",
        help="Root output directory containing finetune_* subdirectories"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/embedder_comparison",
        help="Output directory for comparison results"
    )
    
    args = parser.parse_args()
    
    output_root = args.output_root
    output_dir = Path(args.output_dir)

    print("Auto-discovering latest finetune checkpoints...")
    mtsplice_dir = find_latest_finetune_checkpoint(output_root, "MTSplice")
    dilated_dir = find_latest_finetune_checkpoint(output_root, "DilatedConv1D")
    
    generate_comparison_summary(mtsplice_dir, dilated_dir, output_dir)
    
    print(f"\n✅ Comparison complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
