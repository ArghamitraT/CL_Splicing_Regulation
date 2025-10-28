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
    
    # --- UNCOMMENTED: We need these for the forward selection loop ---
    from scripts.plot_comparison_metrics import (
        logit, get_delta_psi, load_ground_truth, calculate_rmse_by_tissue
    )
except ImportError as e:
    print(f"Error importing project modules (make sure 'scripts.plot_comparison_metrics' is accessible): {e}")
    sys.exit(1)

def get_optimal_num_workers():
    """Calculates optimal num_workers."""
    num_cpus = os.cpu_count()
    num_gpus = torch.cuda.device_count()
    workers = min(num_cpus // max(1, num_gpus), 16) if num_cpus else 0
    return workers

# --- Modular Functions ---

def setup_datamodule_for_validation(config: DictConfig) -> torch.utils.data.DataLoader:
    """Initializes and sets up the DataModule for the validation stage."""
    print("\n--- Setting up DataModule for VALIDATION set ---")
    # --- MODIFIED: Use validation files for this process ---
    # This assumes config.dataset.val_files is correctly set
    config.dataset.test_files = config.dataset.val_files 
    print(f"Using validation files: {config.dataset.test_files}")
    
    data_module = PSIRegressionDataModule(config)
    data_module.setup(stage='test') # Use stage='test' to load the files specified in config.dataset.test_files
    val_dataloader = data_module.test_dataloader() # test_dataloader() reads config.dataset.test_files
    
    if not val_dataloader:
         raise RuntimeError("Validation dataloader not found or empty.")
    print(f"Validation dataset size: {len(val_dataloader.dataset)}")
    return val_dataloader

def identify_ensemble_checkpoints(ensemble_base_path: Path, run_indices: list | None, config: DictConfig) -> dict:
    """Finds and validates checkpoint paths inside the deeper structure."""
    print("\n--- Identifying Ensemble Runs ---")
    if not ensemble_base_path.is_dir():
        raise FileNotFoundError(f"Base experiment path not found: {ensemble_base_path}")

    all_run_dirs = sorted([d for d in ensemble_base_path.iterdir() if d.is_dir() and d.name.startswith('run_')])

    selected_run_dirs = []
    if not run_indices: # Handles None or empty list
        print("Selecting all found run_* directories.")
        selected_run_dirs = all_run_dirs
    else:
        print(f"Selecting specific runs: {run_indices}")
        run_map = {int(d.name.split('_')[-1]): d for d in all_run_dirs}
        for idx in run_indices:
            if idx in run_map:
                selected_run_dirs.append(run_map[idx])
            else:
                print(f"Warning: Run index {idx} specified but directory not found.")

    if not selected_run_dirs:
        raise FileNotFoundError(f"No valid run directories found or selected in {ensemble_base_path}")

    print(f"Found {len(selected_run_dirs)} potential runs:")
    valid_checkpoints = {} 

    model_specific_subdir = Path(config.task._name_) / config.embedder._name_ / str(config.dataset.seq_len)
    print(f"Expecting checkpoints inside subdirectory: {model_specific_subdir}")

    for run_dir in selected_run_dirs:
        model_run_path = run_dir / model_specific_subdir
        ckpt_path = model_run_path / "best-checkpoint.ckpt"

        if ckpt_path.exists():
            print(f"  - {run_dir.name} -> {model_specific_subdir / ckpt_path.name}")
            valid_checkpoints[model_run_path] = str(ckpt_path) 
        else:
            print(f"Warning: Checkpoint not found at {ckpt_path}. Skipping this run.")

    if not valid_checkpoints:
        raise FileNotFoundError("No valid checkpoints found for any selected run.")

    print(f"Identified {len(valid_checkpoints)} models for evaluation.")
    return valid_checkpoints 

def get_model_performance(
    valid_checkpoints: dict,
    config: DictConfig,
    dataloader: torch.utils.data.DataLoader,
    raw_pred_filename: str = "tsplice_raw_output_all_tissues.tsv"
    ) -> list[dict]:
    """
    Runs trainer.test() for each model to get its individual loss and prediction DataFrame.
    Returns a list of dictionaries: [{'loss': float, 'df': pd.DataFrame, 'name': str}, ...]
    """
    print("\n--- Getting Individual Model Performance (Loss + Predictions) ---")
    
    # We create a new trainer and model instance for this process
    trainer = create_trainer(config)
    base_model = initialize_encoders_and_model(config, root_path)
    
    num_ensemble_models = len(valid_checkpoints)
    base_ensemble_output_dir = Path(config.ensemble.output_subdir) / "individual_runs"
    
    model_performance_list = []

    for i, (run_dir, ckpt_path) in enumerate(valid_checkpoints.items()):
        run_name = run_dir.parent.parent.name # e.g., 'run_1'
        print(f"Processing model {i+1}/{num_ensemble_models} ({run_name})...")

        # --- Modify config to set a *unique* output dir for this run ---
        run_specific_output_dir = base_ensemble_output_dir / run_name
        OmegaConf.set_struct(config.ensemble, False) 
        config.ensemble.output_subdir = str(run_specific_output_dir) 
        print(f"  -> Setting output dir for test hook to: {config.ensemble.output_subdir}")
        
        # --- Run trainer.test() and CAPTURE the results ---
        # trainer.test returns a list of dictionaries, one per dataloader
        results = trainer.test(model=base_model, dataloaders=dataloader, ckpt_path=ckpt_path, verbose=False)
        
        # --- Extract the validation loss ---
        try:
            # IMPORTANT: Assumes your test_step logs 'val_loss'. 
            # Change this key if your metric is named differently (e.g., 'test_loss', 'val_rmse').
            individual_loss = results[0]['val_loss'] 
            print(f"  -> Individual Loss (val_loss): {individual_loss:.6f}")
        except (KeyError, IndexError) as e:
            print(f"  -> ERROR: Could not find 'val_loss' in trainer results: {results}. Skipping.")
            continue
            
        # --- Load the prediction file saved by the test hook ---
        pred_file_path = run_specific_output_dir / raw_pred_filename
        if pred_file_path.exists():
            try:
                pred_df = pd.read_csv(pred_file_path, sep='\t')
                if 'exon_id' not in pred_df.columns:
                     print(f"  -> ERROR: 'exon_id' missing in {pred_file_path}. Skipping.")
                     continue
                
                # Store the loss, the prediction DF, and the name
                model_performance_list.append({
                    "loss": individual_loss,
                    "df": pred_df.set_index('exon_id'),
                    "name": run_name
                })
                print(f"  -> Loaded predictions DF shape: {pred_df.shape}")
                
            except Exception as e:
                 print(f"  -> ERROR loading/processing {pred_file_path}: {e}. Skipping.")
        else:
            print(f"  -> ERROR: Prediction file not found at {pred_file_path}. Skipping run.")

    if not model_performance_list:
         raise RuntimeError("Failed to get performance from any model.")
         
    # --- SORT the models by their individual loss ---
    sorted_models = sorted(model_performance_list, key=lambda x: x['loss'])
    
    print("\n--- Model Performance Summary (Sorted by Loss) ---")
    for model_info in sorted_models:
        print(f"  - {model_info['name']}: {model_info['loss']:.6f}")

    return sorted_models


def average_predictions(all_predictions_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Averages predictions across multiple runs, aligning by exon_id (index)."""
    # This function assumes DFs are already indexed by 'exon_id'
    if not all_predictions_dfs:
        raise ValueError("Prediction list is empty, cannot average.")

    combined_preds = pd.concat(all_predictions_dfs, axis=0)
    
    # Group by index (exon_id) and mean all tissue columns
    averaged_predictions_series = combined_preds.groupby(combined_preds.index).mean()
    
    # Convert back to a wide DataFrame
    averaged_preds_df_wide = averaged_predictions_series.reset_index()
    return averaged_preds_df_wide

def save_dataframe(df: pd.DataFrame, save_dir: Path, filename: str):
    """Saves a DataFrame to a specified directory and filename."""
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename
    df.to_csv(save_path, sep='\t', index=False, float_format='%.6f')
    print(f"DataFrame saved to: {save_path}")

def get_common_tissues(pred_df, gt_df):
    """Finds common tissue columns between prediction and ground truth."""
    pred_tissue_cols = pred_df.columns[1:].tolist() # Exclude exon_id
    gt_tissue_cols = gt_df.columns[1:].tolist() # Exclude exon_id
    
    if set(pred_tissue_cols) != set(gt_tissue_cols):
         print("Warning: Tissue names mismatch between predictions and ground truth!")
         common_tissues = sorted(list(set(pred_tissue_cols) & set(gt_tissue_cols)))
         if not common_tissues:
             raise ValueError("No common tissues found. Cannot calculate metrics.")
         print(f"Using {len(common_tissues)} common tissues for evaluation.")
         return common_tissues
    else:
         return pred_tissue_cols

def calculate_ensemble_rmse(
    averaged_preds_df_wide: pd.DataFrame,
    gt_df_full: pd.DataFrame, 
    tissue_cols_for_eval: list[str]
    ) -> float:
    """Calculates and returns the mean RMSE across all common tissues."""
    
    # Filter avg_preds_df to only common tissues
    avg_preds_df_filtered = averaged_preds_df_wide[['exon_id'] + tissue_cols_for_eval]

    try:
        # Calculate Delta PSI for predictions and GT
        delta_psi_pred_df, _ = get_delta_psi(avg_preds_df_filtered, tissue_cols_for_eval, gt_df_full)
        delta_psi_pred_df_long = delta_psi_pred_df.melt(
            id_vars=['exon_id'], value_vars=tissue_cols_for_eval,
            var_name='tissue', value_name='pred_delta_psi'
        )

        # Filter GT df to only common tissues
        delta_psi_gt_df, _ = get_delta_psi(gt_df_full, tissue_cols_for_eval, gt_df_full)
        delta_psi_gt_df = delta_psi_gt_df[['exon_id'] + tissue_cols_for_eval]
        delta_psi_gt_df_long = delta_psi_gt_df.melt(
            id_vars=['exon_id'], value_vars=tissue_cols_for_eval,
            var_name='tissue', value_name='gt_delta_psi'
        )

        # Calculate RMSE
        ensemble_rmse_df = calculate_rmse_by_tissue(delta_psi_pred_df_long, delta_psi_gt_df_long)
        
        # --- Return the single loss value ---
        mean_rmse = ensemble_rmse_df['rmse_delta_psi'].mean()
        return float(mean_rmse)

    except Exception as e:
        print(f"Error calculating ensemble RMSE: {e}")
        return float('inf') # Return a bad score if calculation fails

def calculate_and_save_ensemble_metrics(
    averaged_preds_df_wide: pd.DataFrame,
    gt_df_full: pd.DataFrame,
    tissue_cols_for_eval: list[str],
    save_dir: Path,
    metrics_filename: str
    ):
    """Calculates Delta PSI/Logit and RMSE, then saves the metrics. (Final Report)"""
    print("\n--- Calculating and Saving FINAL Metrics ---")
    
    # Filter avg_preds_df to only common tissues
    avg_preds_df_filtered = averaged_preds_df_wide[['exon_id'] + tissue_cols_for_eval]

    try:
        # Calculate Delta PSI for predictions and GT
        delta_psi_pred_df, _ = get_delta_psi(avg_preds_df_filtered, tissue_cols_for_eval, gt_df_full)
        delta_psi_pred_df_long = delta_psi_pred_df.melt(
            id_vars=['exon_id'], value_vars=tissue_cols_for_eval,
            var_name='tissue', value_name='pred_delta_psi'
        )

        # Filter GT df to only common tissues
        delta_psi_gt_df, _ = get_delta_psi(gt_df_full, tissue_cols_for_eval, gt_df_full)
        delta_psi_gt_df = delta_psi_gt_df[['exon_id'] + tissue_cols_for_eval]
        delta_psi_gt_df_long = delta_psi_gt_df.melt(
            id_vars=['exon_id'], value_vars=tissue_cols_for_eval,
            var_name='tissue', value_name='gt_delta_psi'
        )

        # Calculate RMSE
        ensemble_rmse_df = calculate_rmse_by_tissue(delta_psi_pred_df_long, delta_psi_gt_df_long)

        # Save Metrics
        save_dataframe(ensemble_rmse_df, save_dir, metrics_filename)
        print("Top 5 RMSE results for final ensemble:")
        print(ensemble_rmse_df.sort_values(by='rmse_delta_psi').head())

    except Exception as e:
        print(f"Error calculating or saving final metrics: {e}")


# --- Orchestration Function (Modified for Forward Selection) ---
def main_evaluate_ensemble(config: DictConfig):
    """Orchestrates the ENSEMBLE FORWARD SELECTION strategy."""
    
    # Resolvers
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y, replace=True)
    OmegaConf.register_new_resolver('min', lambda x, y: min(x, y), replace=True)
    OmegaConf.register_new_resolver('device_count', torch.cuda.device_count, replace=True)
    OmegaConf.register_new_resolver('optimal_workers', get_optimal_num_workers, replace=True)

    print("--- Ensemble FORWARD SELECTION Evaluation Script ---")
    print_config(config, resolve=True) 

    results_base_dir = Path(root_path) / "files" / "results"
    ensemble_base_path = results_base_dir / config.ensemble.base_experiment_path
    save_dir = Path(config.ensemble.output_subdir) # Use the full path from config
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Setup Dataloader and Ground Truth (Do this once) ---
    val_dataloader = setup_datamodule_for_validation(config)
    
    gt_file_path = f"{root_path}/Contrastive_Learning/data/final_data/ASCOT_finetuning/{config.dataset.target_file_name}"
    try:
        gt_df_full, gt_tissue_cols = load_ground_truth(gt_file_path)
    except Exception as e:
        print(f"FATAL: Could not load ground truth file at {gt_file_path}: {e}")
        return
        
    # --- 2. Identify all model checkpoints ---
    valid_checkpoints = identify_ensemble_checkpoints(
        ensemble_base_path, config.ensemble.run_indices, config
    )

    # --- 3. Get individual performance and SORT models ---
    sorted_models = get_model_performance(
        valid_checkpoints, config, val_dataloader, config.ensemble.raw_pred_filename
    )

    if not sorted_models:
        print("FATAL: No models were successfully evaluated. Exiting.")
        return

    # --- 4. Forward Selection Loop ---
    print("\n--- Starting Forward Selection Loop ---")
    
    best_ensemble_loss = float('inf')
    best_ensemble_k = 0
    best_ensemble_df = None # This will store the averaged_preds_df of the best ensemble
    
    current_ensemble_dfs = [] # List of DataFrames to be averaged
    
    # We need to determine the common tissues *before* the loop
    # Use the first model's DF to check against GT
    first_model_df = sorted_models[0]['df'].reset_index()
    common_tissues = get_common_tissues(first_model_df, gt_df_full)

    for i, model_info in enumerate(sorted_models):
        k = i + 1
        current_ensemble_dfs.append(model_info['df'])
        
        # Average the predictions of the current K models
        avg_df = average_predictions(current_ensemble_dfs)
        
        # Calculate the loss (mean RMSE) for this K-model ensemble
        current_loss = calculate_ensemble_rmse(avg_df, gt_df_full, common_tissues)
        
        print(f"  Ensemble k={k} (adding {model_info['name']}) -> Loss (Mean RMSE): {current_loss:.6f}")
        
        if current_loss < best_ensemble_loss:
            best_ensemble_loss = current_loss
            best_ensemble_k = k
            best_ensemble_df = avg_df # Save this averaged DataFrame
            print(f"    -> NEW BEST LOSS FOUND!")
        else:
            print(f"    -> Loss did not improve (Current: {current_loss:.6f} vs Best: {best_ensemble_loss:.6f}). Stopping.")
            break # Stop the loop

    # --- 5. Save Final Results ---
    print("\n--- Forward Selection Finished ---")
    
    if best_ensemble_df is None:
        print("FATAL: No valid ensemble was created.")
        return
        
    print(f"Best Ensemble Size (k): {best_ensemble_k}")
    print(f"Best Ensemble Loss (Mean RMSE): {best_ensemble_loss:.6f}")
    
    # Save the predictions of the best ensemble
    save_dataframe(best_ensemble_df, save_dir, config.ensemble.predictions_filename)
    
    # Calculate and save the detailed, per-tissue metrics for the best ensemble
    calculate_and_save_ensemble_metrics(
        best_ensemble_df,
        gt_df_full,
        common_tissues,
        save_dir,
        config.ensemble.metrics_filename
    )

    print("\n--- Ensemble Evaluation Finished Successfully ---")


# --- Main execution block (Using @hydra.main) ---
@hydra.main(version_base=None, config_path="../configs", config_name="psi_regression.yaml")
def main(config: OmegaConf): # Config is loaded by Hydra based on psi_regression.yaml

    # Parameters #
    experiment_folder = "exprmnt_2025_10_22__17_47_17"
    # --- MODIFIED: Output path is now *outside* the checkpoints folder ---
    output_subdir = f"{root_path}/files/results/{experiment_folder}/ensemble_forward_selection_val"

    # --- Define Ensemble Parameters Here ---
    ensemble_params = OmegaConf.create({
        "ensemble": {
            "base_experiment_path": f"{experiment_folder}/weights/checkpoints", 
            
            # --- Auto-detect all runs ---
            # Set this to [] or null to find all 'run_*' folders
            # Or set to [1, 2, 3, ...] to use specific runs
            "run_indices": [], 
            
            "output_subdir": output_subdir, # This is the main *output* dir for this script
            
            # --- Filenames for outputs ---
            "raw_pred_filename": "tsplice_raw_output_all_tissues.tsv",
            "predictions_filename": "ensemble_BEST_predictions.tsv",
            "metrics_filename": "ensemble_BEST_metrics_rmse.tsv"
        }
    })

    # --- Temporarily disable struct mode before merging ---
    OmegaConf.set_struct(config, False)

    # --- Merge ensemble params into the loaded config ---
    config = OmegaConf.merge(config, ensemble_params)

    # --- GPU Selection ---
    def get_free_gpu():
        try:
            result = subprocess.check_output(
                "nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader", shell=True
            )
            memory_used = [int(x) for x in result.decode("utf-8").strip().split("\n")]
            if not memory_used: return 0 
            return memory_used.index(min(memory_used))
        except Exception as e:
            print(f"Warning: Could not query GPU memory ({e}). Defaulting to GPU 0.")
            return 0
            
    if torch.cuda.is_available():
        free_gpu = get_free_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpu)
        print(f"Using GPU {free_gpu}: {torch.cuda.get_device_name(0)}")

    # --- Configure Trainer for INFERENCE (test) mode ---
    config.trainer.max_epochs = 1
    config.trainer.logger = None
    config.trainer.enable_checkpointing = False
    config.aux_models.eval_weights = None
    config.aux_models.train_mode = "eval"
    config.aux_models.warm_start = False
    
    # --- IMPORTANT: Force test_files to be val_files ---
    # The setup_datamodule_for_validation function will now use this
    config.dataset.test_files = config.dataset.val_files 

    # --- Restore struct mode *after* all modifications ---
    OmegaConf.set_struct(config, True)

    # --- Run the main evaluation function with the final config ---
    main_evaluate_ensemble(config)


if __name__ == "__main__":
    main()