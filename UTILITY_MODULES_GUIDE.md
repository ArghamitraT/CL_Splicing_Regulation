# CLADES Utility Modules Guide

Quick reference for the 4 new utility modules created during modularization.

---

## Table of Contents

1. [Training Utilities](#training-utilities)
2. [Metrics Utilities](#metrics-utilities)
3. [Model Utilities](#model-utilities)
4. [Dataset Utilities](#dataset-utilities)
5. [Integration Examples](#integration-examples)

---

## Training Utilities

**File**: `src/utils/training_utils.py`

### Purpose
Consolidate GPU monitoring, epoch tracking, and timing utilities used in model training.

### Functions

#### `log_gpu_memory_stats(log_fn, epoch)`
Logs GPU memory allocation, reservation, and peak memory usage.

**Parameters**:
- `log_fn`: Lightning's `self.log` function or equivalent
- `epoch`: Current epoch number (for reference in logs)

**Behavior**:
- Logs: `gpu_memory_usage`, `gpu_reserved_memory`, `gpu_peak_memory` (in GB)
- Handles CPU-only environments gracefully
- Returns dict with memory stats

**Example**:
```python
from src.utils.training_utils import log_gpu_memory_stats

class MyModel(LightningModule):
    def on_train_epoch_end(self):
        log_gpu_memory_stats(self.log, self.current_epoch)
```

---

#### `log_epoch_timing(log_fn, epoch, epoch_start_time)`
Logs epoch duration with formatted console output.

**Parameters**:
- `log_fn`: Lightning's `self.log` function
- `epoch`: Current epoch number
- `epoch_start_time`: `time.time()` captured at epoch start

**Behavior**:
- Logs: `epoch_time` metric
- Prints formatted message: `"Epoch X took Y.ZZ seconds"`
- Returns epoch duration in seconds

**Example**:
```python
from src.utils.training_utils import log_epoch_timing
import time

class MyModel(LightningModule):
    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()
    
    def on_train_epoch_end(self):
        log_epoch_timing(self.log, self.current_epoch, self.epoch_start_time)
```

---

#### `log_step_timing(log_fn, step, step_start_time, step_name="step")`
Logs individual step (batch) timing.

**Parameters**:
- `log_fn`: Lightning's `self.log` function
- `step`: Step/batch number
- `step_start_time`: `time.time()` at step start
- `step_name`: Optional name for the step (default: "step")

**Behavior**:
- Logs: `{step_name}_time` metric
- Returns step duration in seconds

**Example**:
```python
def training_step(self, batch, batch_idx):
    import time
    step_start = time.time()
    
    # ... training logic ...
    
    log_step_timing(self.log, batch_idx, step_start, "batch")
    return loss
```

---

#### `clear_gpu_cache()`
Clears GPU cache if CUDA is available.

**Behavior**:
- Calls `torch.cuda.empty_cache()` if GPU available
- No-op on CPU-only systems

**Example**:
```python
from src.utils.training_utils import clear_gpu_cache

class MyModel(LightningModule):
    def on_train_epoch_end(self):
        clear_gpu_cache()
```

---

## Metrics Utilities

**File**: `src/utils/metrics_utils.py`

### Purpose
Standardize metric computation (correlation, error metrics) and batch logging.

### Functions

#### `compute_spearman_correlation(y_true, y_pred, nan_policy="propagate")`
Computes Spearman correlation with p-value.

**Parameters**:
- `y_true`: Ground truth values (tensor or array)
- `y_pred`: Predicted values (tensor or array)
- `nan_policy`: How to handle NaN values ("propagate", "omit")

**Returns**:
- Tuple: `(correlation, p_value)`

**Example**:
```python
from src.utils.metrics_utils import compute_spearman_correlation

def on_validation_epoch_end(self):
    corr, pval = compute_spearman_correlation(self.all_preds, self.all_labels)
    self.log("val_spearman", corr)
```

---

#### `compute_pearson_correlation(y_true, y_pred, nan_policy="propagate")`
Computes Pearson correlation with p-value.

**Parameters**: Same as Spearman

**Returns**: Tuple: `(correlation, p_value)`

---

#### `compute_mse(y_true, y_pred)`
Computes Mean Squared Error.

**Parameters**:
- `y_true`: Ground truth values
- `y_pred`: Predicted values

**Returns**: MSE value (float)

**Example**:
```python
from src.utils.metrics_utils import compute_mse

loss = compute_mse(targets, outputs)
```

---

#### `compute_mae(y_true, y_pred)`
Computes Mean Absolute Error.

**Parameters**: Same as MSE

**Returns**: MAE value (float)

---

#### `log_regression_metrics(log_fn, y_true, y_pred, metrics=None)`
Computes and logs multiple regression metrics.

**Parameters**:
- `log_fn`: Lightning's `self.log` function
- `y_true`: Ground truth values
- `y_pred`: Predicted values
- `metrics`: List of metrics to compute (default: `["spearman", "mse", "mae"]`)

**Behavior**:
- Automatically computes requested metrics
- Logs each metric individually
- Handles NaN values gracefully

**Example**:
```python
from src.utils.metrics_utils import log_regression_metrics

def on_validation_epoch_end(self):
    log_regression_metrics(self.log, all_labels, all_preds, 
                          metrics=["spearman", "pearson", "mse"])
```

---

#### `get_tissue_metrics_summary(predictions, labels, tissues)`
Extracts tissue-specific metric summaries.

**Parameters**:
- `predictions`: Model predictions (2D array)
- `labels`: Ground truth labels (2D array)
- `tissues`: List of tissue names (length must match columns)

**Returns**: Dictionary with per-tissue metrics

**Example**:
```python
tissue_metrics = get_tissue_metrics_summary(
    predictions=model_outputs,
    labels=ground_truth,
    tissues=["heart", "brain", "liver"]
)
# Returns: {"heart": {"spearman": 0.95, "mse": 0.01}, ...}
```

---

## Model Utilities

**File**: `src/utils/model_utils.py`

### Purpose
Handle model initialization, encoder management, and weight utilities.

### Functions

#### `get_encoder_output_dimension(encoder)`
Infers the output dimension of an encoder.

**Parameters**:
- `encoder`: Encoder module/model

**Behavior**:
- First tries `encoder.output_dim` attribute
- Falls back to inference using dummy input
- Supports MTSplice and other encoders

**Returns**: Output dimension (integer)

**Example**:
```python
from src.utils.model_utils import get_encoder_output_dimension

encoder = MTSpliceEncoder()
output_dim = get_encoder_output_dimension(encoder)
# Now use output_dim to create downstream layers
```

---

#### `compute_encoder_embeddings(encoder, sequences, batch_size=32)`
Computes embeddings for a batch of sequences.

**Parameters**:
- `encoder`: Encoder module
- `sequences`: Input sequences (tensor or array)
- `batch_size`: Batch size for inference

**Returns**: Embeddings tensor

**Example**:
```python
embeddings = compute_encoder_embeddings(self.encoder, sequences, batch_size=64)
```

---

#### `freeze_encoder_weights(model)`
Freezes all encoder weights (disables gradient updates).

**Parameters**:
- `model`: Model containing encoder

**Behavior**:
- Sets `requires_grad=False` on encoder parameters
- Useful for transfer learning scenarios

**Example**:
```python
from src.utils.model_utils import freeze_encoder_weights

class PSIRegression(LightningModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        freeze_encoder_weights(self)
```

---

#### `unfreeze_encoder_weights(model)`
Unfreezes encoder weights (enables gradient updates).

**Parameters**: Same as freeze

**Example**:
```python
from src.utils.model_utils import unfreeze_encoder_weights

# After initial training phase
unfreeze_encoder_weights(self)
```

---

#### `get_trainable_parameter_count(model)`
Counts trainable parameters in a model.

**Parameters**:
- `model`: PyTorch model

**Returns**: Number of trainable parameters (integer)

**Example**:
```python
from src.utils.model_utils import get_trainable_parameter_count

trainable_params = get_trainable_parameter_count(model)
print(f"Trainable parameters: {trainable_params:,}")
```

---

#### `get_total_parameter_count(model)`
Counts all parameters (trainable + frozen).

**Parameters**: Same as trainable count

**Returns**: Total parameter count (integer)

---

#### `get_model_size_mb(model)`
Estimates model size in megabytes.

**Parameters**: Same as parameter count

**Returns**: Size in MB (float)

**Example**:
```python
from src.utils.model_utils import get_model_size_mb

size = get_model_size_mb(model)
print(f"Model size: {size:.2f} MB")
```

---

## Dataset Utilities

**File**: `src/utils/dataset_utils.py`

### Purpose
Standardize data preprocessing, window extraction, and sequence handling.

### Functions

#### `extract_window_with_padding(sequence, position, window_size, pad_token=0)`
Extracts a window around a position with padding.

**Parameters**:
- `sequence`: Input sequence (array or tensor)
- `position`: Center position for window
- `window_size`: Size of window to extract
- `pad_token`: Value to use for padding

**Returns**: Padded window (same length as window_size)

**Example**:
```python
from src.utils.dataset_utils import extract_window_with_padding

# Extract 400bp window around splice site
window = extract_window_with_padding(seq, splice_position, 400)
```

---

#### `get_windows_with_padding(sequence, donor_pos, acceptor_pos, window_size=400, pad_token=0)`
Extracts multiple windows (donor, acceptor, exon) from a sequence.

**Parameters**:
- `sequence`: Input sequence
- `donor_pos`: Donor splice site position
- `acceptor_pos`: Acceptor splice site position
- `window_size`: Size of each window
- `pad_token`: Padding value

**Returns**: Tuple of (donor_window, acceptor_window, exon_window)

**Behavior**:
- Creates centered windows around each position
- Handles boundary cases with padding
- Consistent with existing CLADES preprocessing

**Example**:
```python
from src.utils.dataset_utils import get_windows_with_padding

donor_w, acceptor_w, exon_w = get_windows_with_padding(
    sequence, donor_pos=200, acceptor_pos=300, window_size=400
)
```

---

#### `normalize_sequence_values(sequence, method="zscore")`
Normalizes sequence values.

**Parameters**:
- `sequence`: Input sequence
- `method`: Normalization method ("zscore", "minmax")

**Returns**: Normalized sequence

**Example**:
```python
normalized_seq = normalize_sequence_values(embedding, method="minmax")
```

---

#### `remove_nan_sequences(sequences, labels, threshold=0.1)`
Filters sequences with excessive NaN values.

**Parameters**:
- `sequences`: Sequence data
- `labels`: Corresponding labels
- `threshold`: Maximum fraction of NaNs allowed

**Returns**: Tuple of (clean_sequences, clean_labels)

**Example**:
```python
clean_seqs, clean_labels = remove_nan_sequences(
    sequences, labels, threshold=0.05
)
```

---

#### `create_data_splits(data, labels, test_size=0.2, val_size=0.1, random_state=None)`
Creates train/val/test splits.

**Parameters**:
- `data`: Input data
- `labels`: Corresponding labels
- `test_size`: Fraction for test set
- `val_size`: Fraction for validation set
- `random_state`: Random seed

**Returns**: Dictionary with `train`, `val`, `test` data and labels

**Example**:
```python
from src.utils.dataset_utils import create_data_splits

splits = create_data_splits(
    data, labels, test_size=0.2, val_size=0.1, random_state=42
)
# Access: splits["train"]["data"], splits["train"]["labels"], etc.
```

---

#### `batch_sequences(sequences, labels, batch_size, shuffle=True)`
Creates batches from sequences.

**Parameters**:
- `sequences`: Input sequences
- `labels`: Labels
- `batch_size`: Batch size
- `shuffle`: Whether to shuffle

**Returns**: Generator yielding batches

**Example**:
```python
for batch_seqs, batch_labels in batch_sequences(seqs, labels, 32):
    # Process batch
    pass
```

---

## Integration Examples

### Example 1: New Model Using Training Utilities

```python
import time
from pytorch_lightning import LightningModule
from src.utils.training_utils import log_gpu_memory_stats, log_epoch_timing, clear_gpu_cache

class NewModel(LightningModule):
    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()
    
    def on_train_epoch_end(self):
        # Clean, concise logging with utilities
        log_epoch_timing(self.log, self.current_epoch, self.epoch_start_time)
        log_gpu_memory_stats(self.log, self.current_epoch)
        clear_gpu_cache()
```

### Example 2: Metric Computation in Validation

```python
from src.utils.metrics_utils import compute_spearman_correlation, compute_mse

def on_validation_epoch_end(self):
    corr, pval = compute_spearman_correlation(
        self.all_preds, self.all_labels
    )
    mse = compute_mse(self.all_preds, self.all_labels)
    
    self.log("val_spearman", corr)
    self.log("val_mse", mse)
```

### Example 3: Using Encoder Utilities

```python
from src.utils.model_utils import (
    get_encoder_output_dimension,
    freeze_encoder_weights,
    get_trainable_parameter_count
)

class TransferLearningModel(LightningModule):
    def __init__(self, pretrained_encoder):
        super().__init__()
        self.encoder = pretrained_encoder
        
        # Infer dimensions automatically
        encoder_dim = get_encoder_output_dimension(self.encoder)
        self.head = nn.Linear(encoder_dim, 1)
        
        # Freeze encoder for transfer learning
        freeze_encoder_weights(self)
        
        # Track trainable params
        print(f"Trainable: {get_trainable_parameter_count(self):,} parameters")
```

### Example 4: Data Processing Pipeline

```python
from src.utils.dataset_utils import (
    get_windows_with_padding,
    remove_nan_sequences,
    create_data_splits
)

# Preprocess data
sequences = [...]  # Your sequences
labels = [...]     # Your labels

# Extract windows
processed = [
    get_windows_with_padding(seq, donor, acceptor)
    for seq, donor, acceptor in zip(sequences, donor_pos, acceptor_pos)
]

# Remove problematic samples
clean_seqs, clean_labels = remove_nan_sequences(processed, labels)

# Create splits
splits = create_data_splits(clean_seqs, clean_labels, random_state=42)

train_data = splits["train"]["data"]
val_data = splits["val"]["data"]
test_data = splits["test"]["data"]
```

---

## Best Practices

### Do's ✓
- Use utility functions for common operations
- Import only what you need: `from src.utils.training_utils import log_gpu_memory_stats`
- Check documentation for optional parameters
- Leverage type hints in utility functions

### Don'ts ✗
- Don't modify utility functions without broad testing
- Don't add model-specific logic to utilities (they're generic)
- Don't ignore NaN handling in metrics
- Don't assume GPU availability (utilities handle gracefully)

---

## Adding New Utilities

To add a new utility function:

1. **Identify the pattern**: Is it used in 2+ places?
2. **Choose the module**: Which category (training, metrics, model, dataset)?
3. **Write the function**: Include docstring with parameters, returns, and examples
4. **Add type hints**: Use `typing` module for clarity
5. **Test thoroughly**: Ensure it works with existing code
6. **Update this guide**: Add function documentation

---

## Questions & Troubleshooting

**Q: Why is my metric NaN?**  
A: Check `nan_policy` in `compute_spearman_correlation()` - may need `nan_policy="omit"`

**Q: Does `log_gpu_memory_stats()` work on CPU?**  
A: Yes, it's GPU-aware and returns zeros on CPU-only systems.

**Q: Can I use utilities in inference?**  
A: Yes! Most utilities are model-agnostic and work in any context.

**Q: How do I update a utility function?**  
A: Make the change in one utility file - all downstream code automatically uses the updated version.

