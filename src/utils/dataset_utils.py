"""
Dataset utilities for CLADES framework.
Consolidates data preprocessing and window extraction logic.
"""

import numpy as np
from typing import Dict, Tuple, Optional


def extract_window_with_padding(sequence: str, 
                               center_pos: int, 
                               window_size: int,
                               pad_char: str = 'N') -> str:
    """
    Extract a window of sequence centered at position, with padding if needed.
    
    Eliminates duplicate window extraction code across dataset files.
    
    Args:
        sequence: Full DNA sequence
        center_pos: Center position for window
        window_size: Size of window (total length)
        pad_char: Character to use for padding
    
    Returns:
        Extracted window (padded if necessary)
    """
    half_size = window_size // 2
    start = center_pos - half_size
    end = start + window_size
    
    seq_start = max(0, start)
    seq_end = min(len(sequence), end)
    window = sequence[seq_start:seq_end]

    left_pad = max(0, -start)
    right_pad = max(0, end - len(sequence))
    
    if left_pad > 0:
        window = pad_char * left_pad + window
    if right_pad > 0:
        window = window + pad_char * right_pad
    
    return window


def get_windows_with_padding(full_sequence: str,
                            acceptor_pos: int,
                            donor_pos: int,
                            exon_start: int,
                            exon_end: int,
                            acceptor_window_size: int = 300,
                            donor_window_size: int = 300,
                            exon_window_size: int = 100) -> Dict[str, str]:
    """
    Extract acceptor, donor, and exon windows from full sequence with padding.
    
    Consolidates multi-window extraction logic.
    
    Args:
        full_sequence: Complete DNA sequence
        acceptor_pos: Position of acceptor site
        donor_pos: Position of donor site
        exon_start: Start position of exon
        exon_end: End position of exon
        acceptor_window_size: Size of acceptor window
        donor_window_size: Size of donor window
        exon_window_size: Size of exon window
    
    Returns:
        Dictionary with keys 'acceptor', 'donor', 'exon' containing extracted windows
    """
    return {
        'acceptor': extract_window_with_padding(
            full_sequence, acceptor_pos, acceptor_window_size
        ),
        'donor': extract_window_with_padding(
            full_sequence, donor_pos, donor_window_size
        ),
        'exon': extract_window_with_padding(
            full_sequence, (exon_start + exon_end) // 2, exon_window_size
        ),
    }


def normalize_sequence_values(values: np.ndarray, 
                             method: str = 'minmax',
                             min_val: float = 0.0,
                             max_val: float = 1.0) -> np.ndarray:
    """
    Normalize sequence or value arrays.
    
    Args:
        values: Input values to normalize
        method: 'minmax' for min-max scaling, 'zscore' for standardization
        min_val: Minimum value for minmax scaling
        max_val: Maximum value for minmax scaling
    
    Returns:
        Normalized values
    """
    if method == 'minmax':
        v_min = np.min(values)
        v_max = np.max(values)
        if v_max == v_min:
            return np.ones_like(values) * min_val
        return min_val + (values - v_min) / (v_max - v_min) * (max_val - min_val)
    
    elif method == 'zscore':
        v_mean = np.mean(values)
        v_std = np.std(values)
        if v_std == 0:
            return np.zeros_like(values)
        return (values - v_mean) / v_std
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def remove_nan_sequences(data: dict, 
                        keys_to_check: Optional[list] = None) -> dict:
    """
    Filter out entries with NaN values in specified keys.
    
    Consolidates NaN handling across datasets.
    
    Args:
        data: Dictionary of data to filter
        keys_to_check: List of keys to check for NaN (if None, check all)
    
    Returns:
        Filtered dictionary with NaN entries removed
    """
    if keys_to_check is None:
        keys_to_check = list(data.keys())

    valid_mask = np.ones(len(data[keys_to_check[0]]), dtype=bool)
    for key in keys_to_check:
        if key in data:
            valid_mask &= ~np.isnan(data[key]).flatten()

    filtered = {}
    for key, values in data.items():
        if isinstance(values, np.ndarray):
            filtered[key] = values[valid_mask]
        elif isinstance(values, list):
            filtered[key] = [v for v, m in zip(values, valid_mask) if m]
        else:
            filtered[key] = values
    
    return filtered


def create_data_splits(data_size: int, 
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      test_ratio: Optional[float] = None,
                      random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create train/val/test split indices.
    
    Consolidates data splitting logic.
    
    Args:
        data_size: Total number of samples
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing (default: 1 - train_ratio - val_ratio)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if test_ratio is None:
        test_ratio = 1.0 - train_ratio - val_ratio

    indices = np.arange(data_size)
    np.random.shuffle(indices)

    n_train = int(train_ratio * data_size)
    n_val = int(val_ratio * data_size)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    return train_idx, val_idx, test_idx


def batch_sequences(sequences: list, 
                   batch_size: int) -> list:
    """
    Group sequences into batches.
    
    Args:
        sequences: List of sequences
        batch_size: Size of each batch
    
    Returns:
        List of batches
    """
    batches = []
    for i in range(0, len(sequences), batch_size):
        batches.append(sequences[i:i + batch_size])
    return batches

