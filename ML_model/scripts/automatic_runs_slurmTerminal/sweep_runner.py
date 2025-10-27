"""
Submit a HYPERPARAMETER SWEEP for PSI regression training on EmpireAI.
"""
import copy
import itertools
import random # Make sure random is imported
import os # Make sure os is imported

# --- Make sure all these functions are in submit_jobs.py ---
from submit_jobs_sweep import (
    setup_paths, 
    create_readme, 
    submit_job, 
    get_file_name, 
    create_bash_file,
    create_slurm_header_psi, # We need this
    create_prg_header_psi    # And thisw
)

# --------------------------------------------------------------------
# 1. BASE CONFIGURATION
# (Parameters that are THE SAME for all sweep runs)
# --------------------------------------------------------------------
def get_base_config():
    """Return all non-swept parameters."""

    cfg = dict(
        slurm_file_name = 'Psi__SWEEP',
        task = "psi_regression_task", # "psi_regression_task" or "introns_cl"
        maxpooling = True,
        global_batch_size = 2048 , # Default, will be overridden by sweep
        max_epochs = 15,
        optimizer = "adam",
        learning_rate =  1e-3,     # Default, will be overridden by sweep
        readme_comment = "SWEEP: new corrected mtsplice model, 300bp, no CL\n",
        wandb_logger_NOTES = "SWEEP new corrected mtsplice model 300bp no CL",
        new_project_wandb = 1, # Set to 1 for sweep logic
        
        ##### --- machine configuration
        gpu_num = 1,
        hour = 3,
        memory = 100,        # GB
        nthred = 8,          # CPUs

        ##### --- embedder specific
        # embedder = "ntv2",
        # tokenizer = "hf_tokenizer",
        # tokenizer_seq_len = 201,
        # embedder = "resnet",
        # tokenizer = "custom_tokenizer",
        # tokenizer_seq_len = 201,
        embedder = "mtsplice",
        tokenizer = "onehot_tokenizer",
        tokenizer_seq_len = 400,
        
        ##### --- CL specific parameters (unused in this task)
        loss_name = "supcon",
        fixed_species = True,
        n_augmentations = 7,

        TRAIN_FILE="train_merged_filtered_min30Views.pkl",
        VAL_FILE="val_merged_filtered_min30Views.pkl",    
        TEST_FILE="test_merged_filtered_min30Views.pkl",
        ##### --- CL specific parameters --- #####

        ##### --- psi specific parameters --- #####
        freeze_encoder = False, # Default, will be overridden by sweep
        warm_start = False,
        psi_loss_name = "MTSpliceBCELoss",
        PSI_TEST_FILE = "psi_variable_Retina___Eye_psi_MERGED.pkl",
        mtsplice_BCE = 1,
        mode = "mtsplice", # mode: or "3p", "5p", "intronOnly", "intronexon", "mtsplice"
        train_mode = "train",
        eval_weights = "exprmnt_2025_08_17__02_17_03", # Unused
        run_num = 20, # Unused
        val_check_interval = 1.0,
        
        ##### --- pretrained weights ---
        ####### mtsplice weights ##########
        # Ignored since warm_start=False, but kept for structure
        mtsplice_weights = "exprmnt_2025_10_22__19_42_17", 
        
        ####### ntv2 weights for 5', 3' and exon (unused) ##########
        weight_3p = "exprmnt_2025_10_20__00_59_06",
        weight_5p = "exprmnt_2025_10_20__01_00_10",
        weight_exon = "exprmnt_2025_10_20__01_00_54",

        ####### resnet weights (unused) ##########
        # ... (other weights commented out) ...
        ##### --- psi specific parameters --- #####
    )
    return cfg

# --------------------------------------------------------------------
# 2. SWEEP GRID CONFIGURATION
# (Parameters you want to iterate over)
# --------------------------------------------------------------------
def define_sweep_grid():
    """
    Define the grid of hyperparameters to sweep over.
    """
    
    # === DEFINE YOUR SWEEP HERE ===
    param_grid = {
        'learning_rate': [1e-3, 5e-4, 1e-4],
        'dropout_rate': [0.1, 0.25, 0.5],
        'global_batch_size': [1024, 2048], # Example: you could add 1024
        'accumulate_grad_batches': [1, 2], # <-- Consider adding this
}

    # param_grid = {
    #         'learning_rate': [1e-3],
    #         'dropout_rate': [0.1],
    #         'global_batch_size': [1024], # Example: you could add 1024
    #         'accumulate_grad_batches': [1], # <-- Consider adding this
    # }
    # ===============================

    # --- This part automatically creates all combinations ---
    keys, values = zip(*param_grid.items())
    run_params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Sweep Grid Defined: {len(run_params_list)} total jobs to submit.")
    return run_params_list

# --------------------------------------------------------------------
# 3. HELPER TO CREATE UNIQUE NAMES
# --------------------------------------------------------------------
def get_run_names(params_dict):
    """Generates a unique run name based on swept params."""
    
    run_suffix = []
    
    # --- This now includes ALL swept parameters ---
    if 'learning_rate' in params_dict:
        run_suffix.append(f"lr_{params_dict['learning_rate']}")
        
    if 'freeze_encoder' in params_dict:
        # This is a key parameter!
        freeze_str = "Freeze" if params_dict['freeze_encoder'] else "Finetune"
        run_suffix.append(freeze_str)
        
    if 'weight_decay' in params_dict and params_dict['weight_decay'] > 0:
        run_suffix.append(f"wd_{params_dict['weight_decay']}")
        
    if 'dropout_rate' in params_dict:
        run_suffix.append(f"do_{params_dict['dropout_rate']}")
    
    # Add any other parameters you sweep (e.g., batch_size)
    if 'global_batch_size' in params_dict:
        run_suffix.append(f"bs_{params_dict['global_batch_size']}")

    if 'accumulate_grad_batches' in params_dict:
        run_suffix.append(f"accum_{params_dict['accumulate_grad_batches']}")

    return "__".join(run_suffix)
# --------------------------------------------------------------------
# 4. MAIN SWEEP RUNNER
# --------------------------------------------------------------------
def main():
    base_cfg = get_base_config()
    sweep_runs = define_sweep_grid()
    paths = setup_paths()

    # This will be the W&B Project name for ALL runs
    wandb_project_name = base_cfg['slurm_file_name']

    for i, run_params in enumerate(sweep_runs):
        print(f"\n--- Submitting Run {i+1}/{len(sweep_runs)} ---")
        
        # 1. Create a fresh config for this run
        run_cfg = copy.deepcopy(base_cfg)
        
        # 2. Update it with the sweep parameters
        run_cfg.update(run_params)

        # 3. Generate unique names for this run
        wandb_run_name = get_run_names(run_params)
        
        # This will be the slurm file name (e.g., SWEEP__lr_0.001__Finetune)
        run_cfg['slurm_file_name'] = f"{base_cfg['slurm_file_name']}__{wandb_run_name}"
        
        # 4. Add the override keys for submit_job to use
        run_cfg['wandb_project_override'] = wandb_project_name
        run_cfg['wandb_run_name_override'] = wandb_run_name
        
        # 5. Update notes to be specific to this run
        run_notes = f"RUN: {wandb_run_name} | {base_cfg['wandb_logger_NOTES']}"
        run_cfg['wandb_logger_NOTES'] = run_notes
        run_cfg['readme_comment'] = f"{run_notes}\nParameters: {run_params}\n"

        # 6. Submit the job
        print(f"  SLURM Job: {run_cfg['slurm_file_name']}")
        print(f"  W&B Run:   {wandb_project_name} / {wandb_run_name}")
        print(f"  Params:    {run_params}")
        
        # We pass the full run_cfg to submit_job
        submit_job(run_cfg, paths)


if __name__ == "__main__":
    main()