"""
Training utilities for CLADES framework.
Consolidates common training patterns: GPU monitoring, epoch tracking, and logging.
"""

import torch
import time
from typing import Optional, Callable


def log_gpu_memory_stats(log_fn: Callable, current_epoch: int) -> None:
    """
    Log GPU memory statistics (allocated, reserved, peak).
    
    Eliminates duplicate GPU memory logging code across model files.
    
    Args:
        log_fn: Lightning logging function (self.log)
        current_epoch: Current training epoch
    """
    if not torch.cuda.is_available():
        return
    
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    peak_memory = torch.cuda.max_memory_allocated(0) / 1e9
    
    log_fn("gpu_memory_allocated_gb", allocated, prog_bar=False, sync_dist=True)
    log_fn("gpu_memory_reserved_gb", reserved, prog_bar=False, sync_dist=True)
    log_fn("gpu_memory_peak_gb", peak_memory, prog_bar=False, sync_dist=True)
    
    print(f"Epoch {current_epoch} - GPU Memory Used: {allocated:.2f} GB, "
          f"Reserved: {reserved:.2f} GB, Peak: {peak_memory:.2f} GB")


def log_epoch_timing(log_fn: Callable, current_epoch: int, epoch_start_time: float) -> None:
    """
    Log the time taken for an epoch.
    
    Consolidates epoch timing logic used in multiple models.
    
    Args:
        log_fn: Lightning logging function (self.log)
        current_epoch: Current training epoch
        epoch_start_time: Time when epoch started (from time.time())
    """
    epoch_time = time.time() - epoch_start_time
    log_fn("epoch_time", epoch_time, prog_bar=True, sync_dist=True)
    print(f"Epoch {current_epoch} took {epoch_time:.2f} seconds.")


def log_step_timing(log_fn: Callable, step_start_time: float, 
                   step_name: str = "step", log_key: str = "step_time") -> float:
    """
    Log the time taken for a single training/validation step.
    
    Args:
        log_fn: Lightning logging function (self.log)
        step_start_time: Time when step started
        step_name: Name of the step for print statements (e.g., "training", "validation")
        log_key: Key to use for logging (default: "step_time")
    
    Returns:
        Step duration in seconds
    """
    step_time = time.time() - step_start_time
    log_fn(log_key, step_time, on_epoch=False, on_step=True, sync_dist=True)
    return step_time


def clear_gpu_cache() -> None:
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

