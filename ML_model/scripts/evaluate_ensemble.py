import sys
import os
import subprocess
import torch
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import hydra 
import pandas as pd
import numpy as np
import pickle  # <-- For loading the results
import logging 
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# *** ADJUST THIS PATH if your file is located elsewhere ***
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.MTSpliceBCE import (
    load_ground_truth, 
    merge_predictions, 
    compute_final_psi, 
    compute_per_tissue_corr, 
    save_and_report 
)
logging.info("Successfully imported evaluation utility functions from MTSpliceBCE module.")


# --- Utility Functions (Copy from your other script) ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def find_contrastive_root(start: Path = Path(__file__)) -> Path:
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

# --- Project-Specific Imports (Copy from your other script) ---
try:
    from src.trainer.utils import create_trainer
    from src.datasets.auxiliary_jobs import PSIRegressionDataModule
    from src.utils.config import print_config
    from src.utils.encoder_init import initialize_encoders_and_model
except ImportError as e:
    print(f"Error importing project modules: {e}")
    sys.exit(1)

def get_optimal_num_workers():
    num_cpus = os.cpu_count()
    num_gpus = torch.cuda.device_count()
    workers = min(num_cpus // max(1, num_gpus), 16) if num_cpus else 0
    return workers

def setup_logging(save_dir: Path, filename: str = "test_evaluation.log"):
    """Configures logging to file and console."""
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = save_dir / filename
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logging.info(f"Logging initialized. Log file: {log_file_path}")

# --- Helper Functions (Copied from your other script) ---

def setup_datamodule_for_test(config: DictConfig) -> torch.utils.data.DataLoader:
    """Initializes and sets up the DataModule for the TEST set."""
    logging.info("\n--- Setting up DataModule for TEST set ---")
    
    # This function assumes config.dataset.test_files is the *correct* test path
    logging.info(f"Using test files: {config.dataset.test_files}")
    
    data_module = PSIRegressionDataModule(config)
    data_module.setup() 
    test_dataloader = data_module.test_dataloader() 
    
    if not test_dataloader:
         raise RuntimeError("Test dataloader not found or empty.")
    logging.info(f"Test dataset size: {len(test_dataloader.dataset)}")
    return test_dataloader

def average_predictions(all_predictions_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Averages predictions across multiple runs, aligning by exon_id (index)."""
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
    logging.info(f"DataFrame saved to: {save_path}")

# --- Core Logic Function (Your simplified function) ---

def get_average_test_prediction(
    best_checkpoints: list[str], 
    config: DictConfig, 
    save_dir: Path,
    raw_pred_filename: str,
    avg_pred_filename: str = "test_ensemble_avg_delta_logit.tsv"
    ):
    """
    Runs the final k-model ensemble on the TEST SET,
    averages the predictions, and saves the result.
    """
    logging.info("\n--- [PHASE 3] Running Final Evaluation on TEST SET ---")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Setup Test Dataloader
    test_dataloader = setup_datamodule_for_test(config)

    # 2. Get Test Set Predictions for the k-best models
    trainer = create_trainer(config)
    base_model = initialize_encoders_and_model(config, root_path)
    test_pred_output_dir = save_dir / "individual_test_runs"
    
    all_test_delta_logit_dfs = []

    for i, ckpt_path in enumerate(best_checkpoints):
        logging.info(f"Getting test preds for model {i+1}/{len(best_checkpoints)}...")
        logging.info(f"  -> Ckpt: {ckpt_path}")
        
        run_specific_output_dir = test_pred_output_dir / f"model_{i+1}"
        OmegaConf.set_struct(config.ensemble, False) 
        config.ensemble.output_subdir = str(run_specific_output_dir) 
        
        trainer.test(model=base_model, dataloaders=test_dataloader, ckpt_path=ckpt_path, verbose=False)

        pred_file_path = run_specific_output_dir / raw_pred_filename
        if pred_file_path.exists():
            try:
                pred_df = pd.read_csv(pred_file_path, sep='\t')
                delta_logit_cols = [col for col in pred_df.columns if col.endswith('_pred_delta_logit')]
                pred_delta_logit_df = pred_df[['exon_id'] + delta_logit_cols].set_index('exon_id')
                pred_delta_logit_df.columns = [
                    col.replace('_pred_delta_logit', '') for col in pred_delta_logit_df.columns
                ]
                all_test_delta_logit_dfs.append(pred_delta_logit_df)
            except Exception as e:
                 logging.error(f"  -> ERROR loading/processing {pred_file_path}: {e}. Skipping.")
        else:
            logging.error(f"  -> ERROR: Prediction file not found at {pred_file_path}. Skipping.")

    if not all_test_delta_logit_dfs:
        logging.critical("FATAL: Could not get any test predictions. Exiting.")
        return

    # 3. Average the Test Set Predictions
    logging.info(f"Averaging test predictions from {len(all_test_delta_logit_dfs)} models...")
    avg_test_delta_logit_df = average_predictions(all_test_delta_logit_dfs)
    # 4. Save the averaged predictions
    save_dataframe(avg_test_delta_logit_df, save_dir, avg_pred_filename)

    # --- Steps after averaging and saving avg_test_delta_logit_df ---
    
    # 4. Prepare inputs for your functions
    logging.info("Preparing inputs for final report functions...")
    # Make sure 'exon_id' is a column if it was reset_index'd during averaging
    if 'exon_id' not in avg_test_delta_logit_df.columns:
        if avg_test_delta_logit_df.index.name == 'exon_id':
             avg_test_delta_logit_df = avg_test_delta_logit_df.reset_index()
        else:
             logging.error("FATAL: 'exon_id' column/index not found in averaged predictions.")
             return # Or sys.exit(1)

    y_pred_ensembled = avg_test_delta_logit_df.drop(columns=['exon_id']).values # NumPy array (N, M)
    exon_ids = avg_test_delta_logit_df['exon_id'].tolist() # List of exon IDs
    pred_shape_M = y_pred_ensembled.shape[1]

    # 5. Load TEST SET ground truth using your function
    try:
        # Assumes config.dataset.test_files holds the original test file path info
        split_name = config.dataset.test_files.intronexon.split('/')[-1].split('_')[1]
        gt_path = os.path.join(config.loss.csv_dir, f"{split_name}_cassette_exons_with_logit_mean_psi.csv")
        # Use your imported load_ground_truth
        gt_df, tissue_cols = load_ground_truth(gt_path, pred_shape_M) 
        # Set index for faster merging if your function doesn't do it
        if gt_df.index.name != 'exon_id':
            gt_df = gt_df.set_index('exon_id')
            
    except Exception as e:
        logging.critical(f"FATAL: Could not load TEST GT file at {gt_path}. {e}", exc_info=True)
        return # Or sys.exit(1)

    # --- 6. Call your imported helper functions sequentially ---
    
    # Merge GT + Averaged Predictions
    # Note: merge_predictions expects gt *without* exon_id index, 
    # but the load_ground_truth above only keeps specific columns. Adjust if needed.
    # Let's assume merge_predictions works correctly with the DataFrame structure.
    # We might need to reset_index on gt_df before merging if merge expects 'exon_id' as a column
    merged_df = merge_predictions(gt_df.reset_index(), exon_ids, y_pred_ensembled, tissue_cols)
    
    # Save the raw/merged file (File 1)
    raw_out_path = save_dir / "tsplice_raw_output_all_tissues.tsv"
    merged_df.to_csv(raw_out_path, sep="\t", index=False, float_format='%.6f')
    logging.info(f"Ensemble raw output (GT + avg_delta_logits) saved to: {raw_out_path}")

    # Compute Final PSI (File 2)
    pred_psi_df = compute_final_psi(merged_df, tissue_cols)
    
    # Compute Metrics (File 3)
    metrics_df = compute_per_tissue_corr(merged_df, pred_psi_df, tissue_cols)
    
    # Save Files 2 and 3 and report
    save_and_report(save_dir, pred_psi_df, metrics_df)
    
    logging.info("\n--- Test Set Evaluation Finished ---")
    
    logging.info(f"\n--- [PHASE 3] Average Test Predictions Saved ---")



def run_ensemble_test_evaluation(
    best_checkpoints_info: list[dict], # List of dicts [{'loss':..,'name':..,'ckpt_path':..}]
    config: DictConfig, 
    save_dir: Path,
    raw_pred_filename: str # Filename like "tsplice_raw_output_all_tissues.tsv"
    ):
    """
    Loads pre-existing raw output files for the k-best models,
    averages the *entire* DataFrame, and generates final reports.
    """
    logging.info("\n--- [PHASE 3] Averaging Pre-Existing Raw Outputs & Generating Final Reports ---")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load the full raw files for the k-best models
    all_raw_dfs = []
         
    for i, model_info in enumerate(best_checkpoints_info):
        # model_name = model_info['name']
        ckpt_path_str = model_info
        ckpt_path = Path(ckpt_path_str) # Convert string to Path object
        run_folder_path = ckpt_path.parents[3] 
        pred_file_path = run_folder_path / raw_pred_filename

        logging.info(f"  -> Loading from: {pred_file_path}")

        if pred_file_path.exists():
            try:
                # Load the full raw file
                raw_df = pd.read_csv(pred_file_path, sep='\t')
                if 'exon_id' not in raw_df.columns:
                    logging.error(f"  -> ERROR: 'exon_id' column missing in {pred_file_path}. Skipping.")
                    continue
                all_raw_dfs.append(raw_df)
                
            except Exception as e:
                 logging.error(f"  -> ERROR loading/processing {pred_file_path}: {e}. Skipping.")
        else:
            logging.error(f"  -> FATAL ERROR: Pre-existing raw prediction file not found at {pred_file_path}.")
            continue # Skip this model if file not found

    if not all_raw_dfs:
        logging.critical("FATAL: Could not load any pre-existing raw files. Exiting.")
        return # Or sys.exit(1)

    # 2. Concatenate and Average the entire DataFrame
    logging.info(f"Averaging {len(all_raw_dfs)} loaded raw DataFrames...")
    
    # Check if all dataframes have the same columns in the same order
    first_cols = all_raw_dfs[0].columns
    for i, df in enumerate(all_raw_dfs[1:], 1):
        if not df.columns.equals(first_cols):
             logging.warning(f"Columns mismatch in file {i+1}. Will attempt to align.")
             # Consider adding alignment logic if needed, otherwise averaging might fail
    
    # Concatenate all dataframes
    concatenated_df = pd.concat(all_raw_dfs, ignore_index=True)
    
    # Group by 'exon_id' and calculate the mean for all numeric columns
    # Non-numeric columns (like exon_id itself, maybe others) will be dropped 
    # unless we handle them separately. 'first()' keeps the first occurrence.
    averaged_merged_df = concatenated_df.groupby('exon_id').agg(
        # Average numeric columns (GT PSI, logit_mean_psi, predictions)
        lambda x: x.mean(skipna=True) if pd.api.types.is_numeric_dtype(x) else x.iloc[0] 
    ).reset_index()

    # Save the averaged raw/merged file (This is your new File 1)
    raw_out_path = save_dir / "tsplice_raw_output_all_tissues_ensembled.tsv"
    save_dataframe(averaged_merged_df, save_dir, "tsplice_raw_output_all_tissues_ensembled.tsv") 

    # 3. Determine Tissue Columns 
    # We need to figure out which columns represent tissues vs predictions vs metadata
    # Re-using logic similar to load_ground_truth on the *averaged* columns
    cols = list(averaged_merged_df.columns)
    # Tissues start after 'logit_mean_psi'
    start_idx = cols.index('logit_mean_psi') + 1
    
    # Find the index of the first prediction column 
    # (assuming they have '_pred_delta' or '_pred_delta_logit')
    first_pred_col_idx = -1
    for i, col_name in enumerate(cols):
        if '_pred_delta' in col_name: # Check for either suffix
            first_pred_col_idx = i
            break
    
    if first_pred_col_idx == -1:
            logging.error("FATAL: Could not find any prediction columns ('_pred_delta' or '_pred_delta_logit') in averaged DataFrame.")
            return None # Indicate failure

    # Tissues end right before the first prediction column
    end_idx = first_pred_col_idx
    
    tissue_cols = cols[start_idx:end_idx]
    
    if not tissue_cols:
            logging.error("FATAL: No tissue columns were identified between 'logit_mean_psi' and the first prediction column.")
            return None # Indicate failure
            
    logging.info(f"Identified {len(tissue_cols)} tissue columns for final metrics (from '{cols[start_idx]}' to '{cols[end_idx-1]}').")
    # --- 4. Call your imported helper functions sequentially ---
    # Pass the averaged_merged_df directly
    
    # Compute Final PSI (File 2) using your function
    pred_psi_df = compute_final_psi(averaged_merged_df, tissue_cols)
    
    # Compute Metrics (File 3) using your function
    metrics_df = compute_per_tissue_corr(averaged_merged_df, pred_psi_df, tissue_cols)
    
    # Save Files 2 and 3 and report using your function
    save_and_report(save_dir, pred_psi_df, metrics_df)
    
    logging.info("\n--- Test Set Evaluation Finished ---")


# --- Main execution block (Using @hydra.main) ---
@hydra.main(version_base=None, config_path="../configs", config_name="psi_regression.yaml")
def main(config: OmegaConf):
    
    # Parameters (MUST MATCH your first script)
    # experiment_folder = "exprmnt_2025_10_28__20_12_11" # intron ofset 300 bp like MTsplice, CL wtdSupcon, MTSplice hyperparameters
    # experiment_folder = "exprmnt_2025_10_28__20_12_58" # intron ofset 300 bp like MTsplice, CL normal Supcon, MTSplice hyperparameters
    # experiment_folder = "exprmnt_2025_10_28__20_28_29" # intron ofset 200 bp like MTsplice, CL normal Supcon, MTSplice hyperparameters
    # experiment_folder = "exprmnt_2025_10_28__20_30_30" # intron ofset 200 bp like MTsplice, CL weighted Supcon, MTSplice hyperparameters
    # experiment_folder = "exprmnt_2025_10_30__13_01_46" # EMPRAIPsi_300bpIntrons_mtspliceHyperparams_2025_10_30__13_01_46
    # experiment_folder = "exprmnt_2025_10_30__14_50_31" # EMPRAIPsi_wtdSupCon_300bpIntrons_mtspliceHyperparams_noExonPadding_2025_10_30__14_50_31
    # experiment_folder = "exprmnt_2025_10_30__14_51_54" # EMPRAIPsi_200bpIntrons_mtspliceHyperparams_noExonPadding_2025_10_30__14_51_54
    experiment_folder = "exprmnt_2025_10_30__14_53_23" # EMPRAIPsi_wtdSupCon_200bpIntrons_mtspliceHyperparams_noExonPadding_2025_10_30__14_53_23
    
    
    # This is the directory where the .pkl file is saved
    output_subdir = f"{root_path}/files/results/{experiment_folder}/ensemble_evaluation_from_valdiation"

    # --- Define Ensemble Parameters Here ---
    ensemble_params = OmegaConf.create({
        "ensemble": {
            "output_subdir": output_subdir,
            "raw_pred_filename": "tsplice_raw_output_all_tissues.tsv",
            # Add any other 'ensemble' keys your config needs
        }
    })

    OmegaConf.set_struct(config, False)
    config = OmegaConf.merge(config, ensemble_params)

    # --- [FIX] ADD THE RESOLVERS HERE ---
    #
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y, replace=True)
    OmegaConf.register_new_resolver('min', lambda x, y: min(x, y), replace=True)
    OmegaConf.register_new_resolver('device_count', torch.cuda.device_count, replace=True)
    OmegaConf.register_new_resolver('optimal_workers', get_optimal_num_workers, replace=True)
    #
    # --- END FIX ---

    # --- Define the *new* save directory for this script ---
    test_save_dir = Path(config.ensemble.output_subdir) / "test_set_evaluation"
    
    # --- SETUP LOGGING ---
    setup_logging(test_save_dir)
    
    logging.info(f"CONTRASTIVE_ROOT set to: {root_path}")
    logging.info(f"Running TEST SET evaluation.")
    logging.info(f"Loading selection results from: {config.ensemble.output_subdir}")

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
            logging.warning(f"Warning: Could not query GPU memory ({e}). Defaulting to GPU 0.")
            return 0
            
    if torch.cuda.is_available():
        free_gpu = get_free_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpu)
        logging.info(f"Using GPU {free_gpu}: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("No GPU found, using CPU.")

    # --- Configure Trainer for INFERENCE (test) mode ---
    config.trainer.max_epochs = 1
    config.trainer.logger = None
    config.trainer.enable_checkpointing = False
    config.aux_models.eval_weights = None
    config.aux_models.train_mode = "eval"
    config.aux_models.warm_start = False
    # config.dataset.test_files.intronexon = "/home/atalukder/Contrastive_Learning/data/final_data/ASCOT_finetuning/psi_test_Retina___Eye_psi_MERGED.pkl"
    
    # --- IMPORTANT ---
    # We *assume* config.dataset.test_files is already the *correct*
    # test file path from your psi_regression.yaml
    logging.info(f"Using original test files: {config.dataset.test_files}")

    OmegaConf.set_struct(config, True)
    
    # --- [NEW] Load the selection results ---
    selection_results_path = Path(config.ensemble.output_subdir) / "ensemble_selection_results.pkl"
    if not selection_results_path.exists():
        logging.critical(f"FATAL: Could not find selection results file at: {selection_results_path}")
        logging.critical("Please run the ensemble selection script first.")
        sys.exit(1)

    try:
        with open(selection_results_path, 'rb') as f:
            selection_results = pickle.load(f)
        
        best_k = selection_results['best_k']
        sorted_models_info = selection_results['sorted_models_info']
        
        logging.info(f"Successfully loaded selection results. Best k = {best_k}")
        
    except Exception as e:
        logging.critical(f"FATAL: Error loading pickle file: {e}", exc_info=True)
        sys.exit(1)
        
    # --- [NEW] Run the test evaluation ---
    
    # 1. Get the list of checkpoint paths for the k-best models
    best_checkpoints = [m['ckpt_path'] for m in sorted_models_info[:best_k]]
    
    # 2. Define the name for the final output file
    avg_test_pred_filename = "test_ensemble_avg_delta_logit.tsv"
    
    # 3. Call the simplified function
    # (AT) DO NOT ERASE:  if you want to validate on test set, then you would have to adjust the folder
    # get_average_test_prediction(
    #     best_checkpoints, 
    #     config, 
    #     test_save_dir, 
    #     config.ensemble.raw_pred_filename,
    #     avg_test_pred_filename
    # )
    # if overhang == 200:
    #     config.dataset.train_files.intronexon = f"{root_path}/data/final_data_old/ASCOT_finetuning/psi_train_Retina___Eye_psi_MERGED.pkl"
    #     config.dataset.val_files.intronexon = f"{root_path}/data/final_data_old/ASCOT_finetuning/psi_val_Retina___Eye_psi_MERGED.pkl"
    #     config.dataset.test_files.intronexon = f"{root_path}/data/final_data_old/ASCOT_finetuning/psi_variable_Retina___Eye_psi_MERGED.pkl"
    # logging.info("\n--- TEST SET SCRIPT FINISHED ---")

    # (AT) DO NOT ERASE: for variable as we already have the prediction, no need to run the trainer.test
    run_ensemble_test_evaluation( # Renamed function
            best_checkpoints, 
            config, 
            test_save_dir, 
            config.ensemble.raw_pred_filename 
        )
    logging.info("\n--- TEST SET SCRIPT FINISHED ---")
        
    


if __name__ == "__main__":
    main()