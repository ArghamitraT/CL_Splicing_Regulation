"""
Metrics utilities for CLADES framework.
Consolidates metric computation and logging across model files.
"""

import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr
from typing import Tuple, Optional, Callable


def compute_spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Compute Spearman correlation coefficient and p-value.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        Tuple of (correlation, p-value)
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) < 2:
        return np.nan, np.nan
    
    return spearmanr(y_true_clean, y_pred_clean)


def compute_pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Compute Pearson correlation coefficient and p-value.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        Tuple of (correlation, p-value)
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) < 2:
        return np.nan, np.nan
    
    return pearsonr(y_true_clean, y_pred_clean)


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute mean squared error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        Mean squared error
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    return np.mean((y_true_clean - y_pred_clean) ** 2)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute mean absolute error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        Mean absolute error
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    return np.mean(np.abs(y_true_clean - y_pred_clean))


def log_regression_metrics(log_fn: Callable, y_true: torch.Tensor, y_pred: torch.Tensor,
                          prefix: str = "", sync_dist: bool = True) -> dict:
    """
    Compute and log multiple regression metrics at once.
    
    Eliminates repeated metric computation across models.
    
    Args:
        log_fn: Lightning logging function (self.log)
        y_true: Ground truth tensor
        y_pred: Predicted tensor
        prefix: Prefix for metric names (e.g., "val_")
        sync_dist: Whether to sync across GPUs
    
    Returns:
        Dictionary of computed metrics
    """
    y_true_np = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
    y_pred_np = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else y_pred

    y_true_np = y_true_np.flatten()
    y_pred_np = y_pred_np.flatten()

    spearman_corr, spearman_pval = compute_spearman_correlation(y_true_np, y_pred_np)
    pearson_corr, pearson_pval = compute_pearson_correlation(y_true_np, y_pred_np)
    mse = compute_mse(y_true_np, y_pred_np)
    mae = compute_mae(y_true_np, y_pred_np)

    metrics = {
        f"{prefix}spearman": spearman_corr,
        f"{prefix}pearson": pearson_corr,
        f"{prefix}mse": mse,
        f"{prefix}mae": mae,
    }
    
    for name, value in metrics.items():
        if not np.isnan(value):
            log_fn(name, value, on_epoch=True, prog_bar=True, sync_dist=sync_dist)
    
    return metrics


def get_tissue_metrics_summary(metrics_dict: dict, tissue_name: str) -> dict:
    """
    Extract tissue-specific metrics from a metrics dictionary.
    
    Args:
        metrics_dict: Dictionary containing all metrics
        tissue_name: Name of tissue to extract
    
    Returns:
        Dictionary with tissue-specific metrics
    """
    tissue_metrics = {}
    for key, value in metrics_dict.items():
        if tissue_name in key:
            tissue_metrics[key] = value
    return tissue_metrics

