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
    data_module.setup()
    val_dataloader = data_module.val_dataloader()
    if not val_dataloader:
         raise RuntimeError("Validation dataloader not found or empty.")
    print(f"Validation dataset size: {len(val_dataloader.dataset)}")
    return val_dataloader



def identify_ensemble_checkpoints(ensemble_base_path: Path, run_indices: list | None, config: DictConfig) -> dict: # Added config
    """Finds and validates checkpoint paths inside the deeper structure."""
    print("\n--- Identifying Ensemble Runs ---")
    if not ensemble_base_path.is_dir():
        raise FileNotFoundError(f"Base experiment path not found: {ensemble_base_path}")

    all_run_dirs = sorted([d for d in ensemble_base_path.iterdir() if d.is_dir() and d.name.startswith('run_')])

    selected_run_dirs = []
    # --- Auto-detect logic ---
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
    # --- End Auto-detect ---

    if not selected_run_dirs:
        raise FileNotFoundError(f"No valid run directories found or selected in {ensemble_base_path}")

    print(f"Found {len(selected_run_dirs)} runs for the ensemble:")
    valid_checkpoints = {} # {run_dir_path: ckpt_path_str}

    # --- Construct the deeper path using config info ---
    # Example: 'psi_regression/MTsplice/201' - adjust if needed
    model_specific_subdir = Path(config.task._name_) / config.embedder._name_ / str(config.dataset.seq_len)
    print(f"Expecting checkpoints inside subdirectory: {model_specific_subdir}")
    # --- End subdirectory construction ---

    for run_dir in selected_run_dirs:
        # --- MODIFIED PATH ---
        model_run_path = run_dir / model_specific_subdir
        ckpt_path = model_run_path / "best-checkpoint.ckpt"
        # --- END MODIFIED PATH ---

        if ckpt_path.exists():
            print(f"  - {run_dir.name} -> {model_specific_subdir / ckpt_path.name}")
            valid_checkpoints[model_run_path] = str(ckpt_path) # Store the deeper path as key now
        else:
            print(f"Warning: Checkpoint not found at {ckpt_path}. Skipping this run.")

    if not valid_checkpoints:
        raise FileNotFoundError("No valid checkpoints found for any selected run.")

    print(f"Using {len(valid_checkpoints)} models in the ensemble.")
    return valid_checkpoints # Keys are now the deeper model_run_path

def generate_predictions_from_checkpoints(
    valid_checkpoints: dict,
    config: DictConfig,
    dataloader: torch.utils.data.DataLoader,
    raw_pred_filename: str = "tsplice_raw_output_all_tissues.tsv"
    ) -> list[pd.DataFrame]:
    """Runs trainer.test() for each checkpoint and loads the saved raw prediction files."""
    print("\n--- Running Validation via trainer.test() and Loading Predictions ---")
    all_predictions_dfs = []

    trainer = create_trainer(config)
    base_model = initialize_encoders_and_model(config, root_path)
    num_ensemble_models = len(valid_checkpoints)
    base_ensemble_output_dir = Path(config.ensemble.output_subdir)

    model_performance_list = []
    for i, (run_dir, ckpt_path) in enumerate(valid_checkpoints.items()):
        print(f"Processing model {i+1}/{num_ensemble_models} from {run_dir.name}...")

        # --- Modify config for this specific run ---
        run_specific_output_dir = base_ensemble_output_dir / f'run_{str(i+1)}'
        # Ensure OmegaConf allows modification even if struct was restored
        OmegaConf.set_struct(config.ensemble, False) 
        config.ensemble.output_subdir = str(run_specific_output_dir) 
        print(f"  -> Setting output dir for test hook to: {config.ensemble.output_subdir}")
        # OmegaConf.set_struct(config.ensemble, True) # Optional: restore struct

        results = trainer.test(model=base_model, dataloaders=dataloader, ckpt_path=ckpt_path)
        # --- Extract the validation loss ---
        try:
            # IMPORTANT: Assumes your test_step logs 'test_loss_epoch'. 
            # Change this key if your metric is named differently (e.g., 'test_loss', 'val_rmse').
            individual_loss = results[0]['test_loss_epoch'] 
            print(f"  -> Individual Loss (val_loss): {individual_loss:.6f}")
        except (KeyError, IndexError) as e:
            print(f"  -> ERROR: Could not find 'val_loss' in trainer results: {results}. Skipping.")
            continue
            
        pred_file_path = run_specific_output_dir / raw_pred_filename
        if pred_file_path.exists():
            try:
                # Load the full file
                pred_df = pd.read_csv(pred_file_path, sep='\t')
                
                # --- NEW: Extract only the delta_logit columns ---
                delta_logit_cols = [col for col in pred_df.columns if col.endswith('_pred_delta_logit')]
                
                if not delta_logit_cols:
                    print(f"  -> ERROR: No '_pred_delta_logit' columns found in {pred_file_path}. Skipping.")
                    continue

                # --- NEW: Create a new DF with just exon_id and delta_logits ---
                pred_delta_logit_df = pred_df[['exon_id'] + delta_logit_cols].set_index('exon_id')
                
                # Rename columns to remove suffix for easier matching
                pred_delta_logit_df.columns = [
                    col.replace('_pred_delta_logit', '') for col in pred_delta_logit_df.columns
                ]
                
                model_performance_list.append({
                    "loss": individual_loss,
                    "df": pred_delta_logit_df, # Store the delta_logit_df
                    "name": f"run_{str(i+1)}"
                })
                print(f"  -> Loaded & processed delta_logits DF shape: {pred_delta_logit_df.shape}")
                
            except Exception as e:
                 print(f"  -> ERROR loading/processing {pred_file_path}: {e}. Skipping.")
        else:
            print(f"  -> ERROR: Prediction file not found at {pred_file_path}. Skipping run.")

    if not model_performance_list:
         raise RuntimeError("Failed to get performance from any model.")
         
    sorted_models = sorted(model_performance_list, key=lambda x: x['loss'])
    
    print("\n--- Model Performance Summary (Sorted by Loss) ---")
    for model_info in sorted_models:
        print(f"  - {model_info['name']}: {model_info['loss']:.6f}")

    return sorted_models
    
def average_predictions(all_predictions_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Averages predictions across multiple runs, aligning by exon_id (index)."""
    # This function now correctly averages the delta-logit DataFrames
    if not all_predictions_dfs:
        raise ValueError("Prediction list is empty, cannot average.")

    combined_preds = pd.concat(all_predictions_dfs, axis=0)
    averaged_predictions_series = combined_preds.groupby(combined_preds.index).mean()
    averaged_preds_df_wide = averaged_predictions_series.reset_index()
    return averaged_preds_df_wide


def save_dataframe(df: pd.DataFrame, save_dir: Path, filename: str):
    """Saves a DataFrame to a specified directory and filename."""
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename
    df.to_csv(save_path, sep='\t', index=False, float_format='%.6f')
    print(f"DataFrame saved to: {save_path}")

def calculate_and_save_ensemble_metrics(
    averaged_delta_logit_df: pd.DataFrame, # This is now avg delta_logits
    gt_lookup_df: pd.DataFrame, # This is the GT file with 'logit_mean_psi'
    gt_df_for_metrics: pd.DataFrame, # This is the full metrics GT file
    common_tissues: list[str], # The list of common tissue names
    save_dir: Path,
    metrics_filename: str
    ):
    """
    Converts avg_delta_logits to psi_pred, then calculates and saves 
    the final Delta PSI/Logit and RMSE metrics.
    """
    print("\n--- Calculating and Saving FINAL Metrics ---")
    
    try:
        # --- NEW: Convert avg_delta_logits to avg_psi_pred ---
        
        # 1. Align pred and gt_lookup
        pred_df_indexed = averaged_delta_logit_df.set_index('exon_id')
        pred_aligned, gt_aligned = pred_df_indexed.align(
            gt_lookup_df, join='inner', axis=0
        )
        
        # 2. Get matching logit_mean_psi
        logit_mean_psi = torch.tensor(
            gt_aligned["logit_mean_psi"].values,
            dtype=torch.float32
        ) # Shape (B,)
        
        # 3. Get avg_delta_logit tensor
        avg_delta_logit_tensor = torch.tensor(
            pred_aligned[common_tissues].values,
            dtype=torch.float32
        ) # Shape (B, 56)
        
        # 4. Convert delta_logits -> logits -> psi
        logits_tensor = avg_delta_logit_tensor + logit_mean_psi[:, None]
        avg_psi_tensor = torch.sigmoid(logits_tensor)
        
        # 5. Convert back to DataFrame for metric functions
        averaged_preds_df_wide = pd.DataFrame(
            avg_psi_tensor.numpy(),
            columns=common_tissues
        )
        averaged_preds_df_wide['exon_id'] = pred_aligned.index
        # --- END CONVERSION ---

        # The rest of the function now works as before
        avg_preds_df_filtered = averaged_preds_df_wide[['exon_id'] + common_tissues]

        # Calculate Delta PSI for predictions and GT
        delta_psi_pred_df, _ = get_delta_psi(avg_preds_df_filtered, common_tissues, gt_df_for_metrics)
        delta_psi_pred_df_long = delta_psi_pred_df.melt(
            id_vars=['exon_id'], value_vars=common_tissues,
            var_name='tissue', value_name='pred_delta_psi'
        )

        gt_df_filtered = gt_df_for_metrics[['exon_id'] + common_tissues + ['mean_psi']] 
        delta_psi_gt_df, _ = get_delta_psi(gt_df_filtered, common_tissues, gt_df_filtered)
        
        delta_psi_gt_df_long = delta_psi_gt_df.melt(
            id_vars=['exon_id'], value_vars=common_tissues,
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

    
    val_dataloader = setup_datamodule_for_validation(config)
    valid_checkpoints = identify_ensemble_checkpoints(
        ensemble_base_path, config.ensemble.run_indices, config
    )
    sorted_models = generate_predictions_from_checkpoints(
        valid_checkpoints, config, val_dataloader
    )
    if not sorted_models:
        print("FATAL: No models were successfully evaluated. Exiting.")
        return
    
    # --- 6. Forward Selection Loop ---
    print(f"\n--- Starting Forward Selection Loop (Metric: {config.loss._target_}) ---")
    
    best_ensemble_loss = float('inf')
    best_ensemble_k = 0
    best_ensemble_df = None # This will store the *averaged_delta_logit_df*
    
    current_ensemble_dfs = [] # List of delta-logit DataFrames
    
    # Determine common tissues before the loop
    # We use the columns from the first *delta_logit* DF
    common_tissues = sorted_models[0]['df'].columns.tolist()

    try:
        loss_fn = hydra.utils.instantiate(config.loss, _recursive_=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_fn.to(device) # Move loss function to device
        print(f"Successfully instantiated loss function: {config.loss._target_}")
    except Exception as e:
        print(f"FATAL: Could not instantiate loss function: {e}")
        return
    

    for i, model_info in enumerate(sorted_models):
        k = i + 1
        current_ensemble_dfs.append(model_info['df'])
        
        # Average the 'delta_logit' values
        avg_delta_logit_df = average_predictions(current_ensemble_dfs)
        
        # --- Convert averaged_delta_logit_df to tensor ---
        # Align with GT to get exon_ids in the correct order for the loss_fn
        # pred_aligned, _ = avg_delta_logit_df.set_index('exon_id').align(
        #     gt_val_df, join='inner', axis=0
        # )
        
        # if pred_aligned.empty:
        #     print(f"  -> Skipping k={k}, no common exons with GT file.")
        #     continue

        pred_aligned = avg_delta_logit_df.set_index('exon_id')
            
        exon_ids = pred_aligned.index.tolist()
        delta_logits_tensor = torch.tensor(
            pred_aligned[common_tissues].values,
            dtype=torch.float32,
            device=device # Move tensor to the same device as the loss_fn
        )

        # --- Call your *actual* loss function ---
        loss_tensor = loss_fn(delta_logits_tensor, exon_ids, split='val')
        current_loss = loss_tensor.item()
        
        print(f"  Ensemble k={k} (adding {model_info['name']}) -> Loss (from class): {current_loss:.6f}")
        
        if current_loss < best_ensemble_loss:
            best_ensemble_loss = current_loss
            best_ensemble_k = k
            best_ensemble_df = avg_delta_logit_df # Save the averaged delta-logit DF
            print(f"    -> NEW BEST LOSS FOUND!")
        else:
            print(f"    -> Loss did not improve (Current: {current_loss:.6f} vs Best: {best_ensemble_loss:.6f}). Stopping.")
            break 

    # --- 7. Save Final Results ---
    print("\n--- Forward Selection Finished ---")
    
    if best_ensemble_df is None:
        print("FATAL: No valid ensemble was created.")
        return
        
    print(f"Best Ensemble Size (k): {best_ensemble_k}")
    print(f"Best Ensemble Loss (from class): {best_ensemble_loss:.6f}")
    
    # Save the predictions (averaged DELTA LOGITS) of the best ensemble
    save_dataframe(best_ensemble_df, save_dir, config.ensemble.predictions_filename)
    
    # Calculate and save the final RMSE metrics
    # calculate_and_save_ensemble_metrics(
    #     best_ensemble_df,      # The avg_delta_logit_df
    #     gt_val_df,             # The GT file for logit_mean_psi lookup
    #     gt_df_for_metrics,     # The full GT file for final metrics
    #     common_tissues,
    #     save_dir,
    #     config.ensemble.metrics_filename
    # )

    print("\n--- Ensemble Evaluation Finished Successfully ---")



# --- Main execution block (Using @hydra.main) ---
@hydra.main(version_base=None, config_path="../configs", config_name="psi_regression.yaml")
def main(config: OmegaConf): # Config is loaded by Hydra based on psi_regression.yaml


    # Parameters #
    experiment_folder = "exprmnt_2025_10_22__17_47_17"
    output_subdir = f"{root_path}/files/results/{experiment_folder}/ensemble_evaluation_from_valdiation"

    # --- Define Ensemble Parameters Here ---
    ensemble_params = OmegaConf.create({
        "ensemble": {
            # --- CORRECTED PATH ---
            "base_experiment_path": f"{experiment_folder}/weights/checkpoints", # <<< Stops at checkpoints
            # --- OPTION 1: Specify runs ---
            # "run_indices": [1, 2, 3, 4, 5, 6, 7, 8], # <<< Keep if you want specific runs
            # --- OPTION 2: Auto-detect all runs ---
            "run_indices": [], # <<< Use empty list or null to find all run_* folders
            "output_subdir": output_subdir,
            "predictions_filename": "tsplice_ensemble_avg_delta_logit_all_tissues.tsv",
        }
    })

    # --- FIX: Temporarily disable struct mode before merging ---
    OmegaConf.set_struct(config, False)

    # --- Merge ensemble params into the loaded config ---
    config = OmegaConf.merge(config, ensemble_params)

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

    config.trainer.max_epochs = 1
    config.trainer.logger = None
    config.trainer.enable_checkpointing = False
    config.aux_models.eval_weights = None
    config.aux_models.train_mode = "eval"
    config.aux_models.warm_start = False
    config.dataset.test_files = config.dataset.val_files

    # --- Optional: Restore struct mode *after* all modifications ---
    OmegaConf.set_struct(config, True)

    # --- Run the main evaluation function with the final config ---
    main_evaluate_ensemble(config)


if __name__ == "__main__":
    main()