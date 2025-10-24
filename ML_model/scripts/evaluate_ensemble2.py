import sys
import os
import subprocess
import torch
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import hydra # Import hydra for the decorator
import pandas as pd
import numpy as np
import importlib
import time
import shutil

# --- Utility Functions (Keep these) ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def find_contrastive_root(start: Path = Path(__file__)) -> Path:
    """Finds the 'Contrastive_Learning' root directory."""
    for parent in start.resolve().parents:
        if parent.name == "Contrastive_Learning":
            return parent
    raise RuntimeError("Could not find 'Contrastive_Learning' directory.")

try:
    root_path = str(find_contrastive_root())
    os.environ["CONTRASTIVE_ROOT"] = root_path
    print(f"CONTRASTIVE_ROOT set to: {root_path}")
except RuntimeError as e:
    print(f"Error finding CONTRASTIVE_ROOT: {e}.")
    sys.exit(1)

try:
    from src.trainer.utils import create_trainer
    from src.datasets.auxiliary_jobs import PSIRegressionDataModule
    from src.utils.config import print_config
    from src.utils.encoder_init import initialize_encoders_and_model
    # from scripts.plot_comparison_metrics import (
    #     logit, get_delta_psi, load_ground_truth, calculate_rmse_by_tissue
    # )
except ImportError as e:
    print(f"Error importing project modules: {e}")
    sys.exit(1)

def get_optimal_num_workers():
    """Calculates optimal num_workers."""
    num_cpus = os.cpu_count()
    num_gpus = torch.cuda.device_count()
    workers = min(num_cpus // max(1, num_gpus), 16) if num_cpus else 0
    return workers

# --- Modular Functions (Keep these defined globally) ---
# setup_datamodule_for_validation
# identify_ensemble_checkpoints
# generate_predictions_from_checkpoints
# average_predictions
# save_dataframe
# calculate_and_save_ensemble_metrics
# (Definitions omitted for brevity - use the ones from the previous response)

def setup_datamodule_for_validation(config: DictConfig) -> torch.utils.data.DataLoader:
    """Initializes and sets up the DataModule for the validation stage."""
    print("\n--- Setting up DataModule for VALIDATION set ---")
    data_module = PSIRegressionDataModule(config)
    data_module.setup(stage="validate")
    val_dataloader = data_module.val_dataloader()
    if not val_dataloader:
         raise RuntimeError("Validation dataloader not found or empty.")
    print(f"Validation dataset size: {len(val_dataloader.dataset)}")
    return val_dataloader

def identify_ensemble_checkpoints(ensemble_base_path: Path, run_indices: list | None) -> dict:
    """Finds and validates checkpoint paths for specified run indices."""
    print("\n--- Identifying Ensemble Runs ---")
    if not ensemble_base_path.is_dir():
        raise FileNotFoundError(f"Base experiment path not found: {ensemble_base_path}")

    all_run_dirs = sorted([d for d in ensemble_base_path.iterdir() if d.is_dir() and d.name.startswith('run_')])

    selected_run_dirs = []
    if run_indices and len(run_indices) > 0:
        print(f"Selecting specific runs: {run_indices}")
        run_map = {int(d.name.split('_')[-1]): d for d in all_run_dirs}
        for idx in run_indices:
            if idx in run_map:
                selected_run_dirs.append(run_map[idx])
            else:
                print(f"Warning: Run index {idx} specified but directory not found.")
    else:
        print("Selecting all found runs.")
        selected_run_dirs = all_run_dirs

    if not selected_run_dirs:
        raise FileNotFoundError(f"No valid run directories found or selected in {ensemble_base_path}")

    print(f"Found {len(selected_run_dirs)} runs for the ensemble:")
    valid_checkpoints = {} # {run_dir_path: ckpt_path_str}
    for run_dir in selected_run_dirs:
        ckpt_path = run_dir / "best-checkpoint.ckpt"
        if ckpt_path.exists():
            print(f"  - {run_dir.name} -> {ckpt_path.name}")
            valid_checkpoints[run_dir] = str(ckpt_path)
        else:
            print(f"Warning: Checkpoint not found in {run_dir}. Skipping this run.")

    if not valid_checkpoints:
        raise FileNotFoundError("No valid checkpoints found for any selected run.")

    print(f"Using {len(valid_checkpoints)} models in the ensemble.")
    return valid_checkpoints

def generate_predictions_from_checkpoints(
    valid_checkpoints: dict,
    config: DictConfig,
    dataloader: torch.utils.data.DataLoader,
    raw_pred_filename: str = "tsplice_raw_output_all_tissues.tsv"
    ) -> list[pd.DataFrame]:
    """Runs trainer.test() for each checkpoint and loads the saved raw prediction files."""
    print("\n--- Running Validation via trainer.test() and Loading Predictions ---")
    all_predictions_dfs = []

    trainer = create_trainer(config, prediction_mode=True)
    base_model = initialize_encoders_and_model(config, root_path).float()
    num_ensemble_models = len(valid_checkpoints)

    for i, (run_dir, ckpt_path) in enumerate(valid_checkpoints.items()):
        print(f"Processing model {i+1}/{num_ensemble_models} from {run_dir.name}...")

        try:
             trainer.test(model=base_model, dataloaders=dataloader, ckpt_path=ckpt_path)
        except Exception as e:
            print(f"  -> ERROR during trainer.test: {e}. Skipping run.")
            continue

        pred_file_path = run_dir / raw_pred_filename
        if pred_file_path.exists():
            try:
                pred_df = pd.read_csv(pred_file_path, sep='\t')
                if 'exon_id' not in pred_df.columns:
                     print(f"  -> ERROR: 'exon_id' missing in {pred_file_path}. Skipping.")
                     continue
                all_predictions_dfs.append(pred_df.set_index('exon_id'))
                print(f"  -> Loaded predictions DF shape: {pred_df.shape}")
            except Exception as e:
                 print(f"  -> ERROR loading/processing {pred_file_path}: {e}. Skipping.")
        else:
            print(f"  -> ERROR: Prediction file not found at {pred_file_path}. Skipping run.")

    if not all_predictions_dfs:
         raise RuntimeError("Failed to load predictions from any run's output file.")

    return all_predictions_dfs

def average_predictions(all_predictions_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Averages predictions across multiple runs, aligning by exon_id."""
    print("\n--- Averaging Predictions ---")
    if not all_predictions_dfs:
        raise ValueError("Prediction list is empty, cannot average.")

    combined_preds = pd.concat(all_predictions_dfs, axis=0)
    averaged_predictions_series = combined_preds.groupby(combined_preds.index).mean()
    averaged_preds_df_wide = averaged_predictions_series.reset_index()
    print(f"Averaged predictions DataFrame shape: {averaged_preds_df_wide.shape}")
    return averaged_preds_df_wide

def save_dataframe(df: pd.DataFrame, save_dir: Path, filename: str):
    """Saves a DataFrame to a specified directory and filename."""
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename
    df.to_csv(save_path, sep='\t', index=False, float_format='%.6f')
    print(f"DataFrame saved to: {save_path}")

def calculate_and_save_ensemble_metrics(
    averaged_preds_df_wide: pd.DataFrame,
    gt_config: DictConfig, # Pass config section relevant to GT
    save_dir: Path,
    metrics_filename: str
    ):
    """Calculates Delta PSI/Logit and RMSE, then saves the metrics."""
    print("\n--- Calculating and Saving Metrics ---")

    # Determine Ground Truth File Path from config
    gt_file_path = f"{root_path}/Contrastive_Learning/data/final_data/ASCOT_finetuning/{gt_config.target_file_name}"

    try:
        gt_df_full, gt_tissue_cols = load_ground_truth(gt_file_path)
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {gt_file_path}")
        return # Or raise
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return # Or raise

    pred_tissue_cols = averaged_preds_df_wide.columns[1:].tolist() # Exclude exon_id

    # Check tissue consistency
    if set(pred_tissue_cols) != set(gt_tissue_cols):
         print("Warning: Tissue names mismatch between predictions and ground truth!")
         common_tissues = sorted(list(set(pred_tissue_cols) & set(gt_tissue_cols)))
         if not common_tissues:
             print("Error: No common tissues found. Cannot calculate metrics.")
             return
         print(f"Using {len(common_tissues)} common tissues for evaluation.")
         tissue_cols_for_eval = common_tissues
         avg_preds_df_filtered = averaged_preds_df_wide[['exon_id'] + tissue_cols_for_eval]
    else:
         tissue_cols_for_eval = pred_tissue_cols
         avg_preds_df_filtered = averaged_preds_df_wide

    try:
        # Calculate Delta PSI for predictions and GT
        delta_psi_pred_df, _ = get_delta_psi(avg_preds_df_filtered, tissue_cols_for_eval, gt_df_full)
        delta_psi_pred_df_long = delta_psi_pred_df.melt(
            id_vars=['exon_id'], value_vars=tissue_cols_for_eval,
            var_name='tissue', value_name='pred_delta_psi'
        )

        delta_psi_gt_df, _ = get_delta_psi(gt_df_full, gt_tissue_cols, gt_df_full)
        if set(pred_tissue_cols) != set(gt_tissue_cols):
            delta_psi_gt_df = delta_psi_gt_df[['exon_id'] + tissue_cols_for_eval]
        delta_psi_gt_df_long = delta_psi_gt_df.melt(
            id_vars=['exon_id'], value_vars=tissue_cols_for_eval,
            var_name='tissue', value_name='gt_delta_psi'
        )

        # Calculate RMSE
        ensemble_rmse_df = calculate_rmse_by_tissue(delta_psi_pred_df_long, delta_psi_gt_df_long)

        # Save Metrics
        save_dataframe(ensemble_rmse_df, save_dir, metrics_filename)
        print("Top 5 RMSE results:")
        print(ensemble_rmse_df.sort_values(by='rmse_delta_psi').head())

    except Exception as e:
        print(f"Error calculating or saving metrics: {e}")


# --- Orchestration Function (Keep this defined globally) ---
def main_evaluate_ensemble(config: DictConfig):
    """Orchestrates the ensemble evaluation using modular functions."""
    
    # Resolvers (important if base config uses them)
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y, replace=True)
    OmegaConf.register_new_resolver('min', lambda x, y: min(x, y), replace=True)
    OmegaConf.register_new_resolver('device_count', torch.cuda.device_count, replace=True)
    OmegaConf.register_new_resolver('optimal_workers', get_optimal_num_workers, replace=True)

    print("--- Ensemble Evaluation Script ---")
    print_config(config, resolve=True) # Print the final config

    results_base_dir = Path(root_path) / "files" / "results"
    # Read ensemble path from the config object
    ensemble_base_path = results_base_dir / config.ensemble.base_experiment_path
    save_dir = ensemble_base_path / config.ensemble.output_subdir

    try:
        val_dataloader = setup_datamodule_for_validation(config)
        valid_checkpoints = identify_ensemble_checkpoints(
            ensemble_base_path, OmegaConf.to_object(config.ensemble.run_indices)
        )
        all_predictions_dfs = generate_predictions_from_checkpoints(
            valid_checkpoints, config, val_dataloader
        )
        averaged_preds_df = average_predictions(all_predictions_dfs)
        save_dataframe(averaged_preds_df, save_dir, config.ensemble.predictions_filename)
        calculate_and_save_ensemble_metrics(
            averaged_preds_df,
            config.data,
            save_dir,
            config.ensemble.metrics_filename
        )
    except (FileNotFoundError, RuntimeError, ValueError, KeyError, Exception) as e:
        print(f"\n--- SCRIPT FAILED ---")
        print(f"Error: {e}")
        sys.exit(1)

    print("\n--- Ensemble Evaluation Finished Successfully ---")


# --- Main execution block (Using @hydra.main) ---
@hydra.main(version_base=None, config_path="../configs", config_name="psi_regression.yaml")
def main(config: OmegaConf): # Config is loaded by Hydra based on psi_regression.yaml

    # --- Define Ensemble Parameters Here ---
    # These will be MERGED into the config loaded by Hydra
    ensemble_params = OmegaConf.create({
        "ensemble": {
            # --- USER INPUT NEEDED HERE ---
            "base_experiment_path": "exprmnt_2025_10_22__17_47_17/weights/checkpoints/psi_regression/MTsplice/201", # <<< UPDATE THIS if needed
            "run_indices": [1, 2, 3, 4, 5, 6, 7, 8], # <<< UPDATE THIS if needed
            # "run_indices": [], # Use empty list or None for all runs
            "output_subdir": "ensemble_evaluation_from_test",
            "metrics_filename": "ensemble_metrics_by_tissue.tsv",
            "predictions_filename": "ensemble_raw_predictions_all_tissues.tsv"
        }
    })

    # --- FIX: Temporarily disable struct mode before merging ---
    OmegaConf.set_struct(config, False)

    # --- Merge ensemble params into the loaded config ---
    config = OmegaConf.merge(config, ensemble_params)

    # --- Optional: Restore struct mode after merging ---
    OmegaConf.set_struct(config, True)
    print_config(config, resolve=True)

    # --- GPU Selection (Copied from your example) ---
    def get_free_gpu():
        try:
            result = subprocess.check_output(
                "nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader", shell=True
            )
            memory_used = [int(x) for x in result.decode("utf-8").strip().split("\n")]
            if not memory_used: return 0 # Default to GPU 0 if no GPUs found
            return memory_used.index(min(memory_used))
        except Exception as e:
            print(f"Warning: Could not query GPU memory ({e}). Defaulting to GPU 0.")
            return 0
            
    if torch.cuda.is_available():
        free_gpu = get_free_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpu)
        print(f"Using GPU {free_gpu}: {torch.cuda.get_device_name(0)}")

    # --- Run the main evaluation function with the final config ---
    main_evaluate_ensemble(config)


if __name__ == "__main__":
    main()