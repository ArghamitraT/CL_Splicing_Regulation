"""
Submit ntv2 PSI regression training on EmpireAI cluster.
Modularized version for clarity and reuse.
"""


from submit_jobs import setup_paths, create_readme, submit_job


# --------------------------------------------------------------------
# 1. CONFIGURATION BLOCK
# --------------------------------------------------------------------
def get_experiment_config():
    """Return all experiment-level parameters (edit here only)."""

    cfg = dict(
        slurm_file_name = 'CL_OOM_trial',
        task = "introns_cl", # "psi_regression_task" or "introns_cl"
        maxpooling = True,
        global_batch_size = 8192,
        max_epochs = 25,
        optimizer = "adam",
        learning_rate =  1e-3,
        readme_comment = "CL trying to check OOM\n",
        wandb_logger_NOTES = "Cl trying to check OOM",
        new_project_wandb = 0,
        fivep_ovrhang = 300,
        threep_ovrhang = 300,
        
        ##### --- machine configuration
        gpu_num = 1,
        hour = 5,
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
        
        ################### --- CL specific parameters ################### 
        loss_name = "weighted_supcon",
        fixed_species = False,
        n_augmentations = 10,
        temperature     = 0.2,
        accumulate_grad_batches = 2,
    
        TRAIN_FILE="train_merged_filtered_min30Views.pkl",
        VAL_FILE="val_merged_filtered_min30Views.pkl",    
        TEST_FILE="test_merged_filtered_min30Views.pkl",

        
        ################### --- CL specific parameters ################### 

        ################### --- psi specific parameters ################### 
        freeze_encoder = False,
        warm_start = False,
        psi_loss_name = "MTSpliceBCELoss",
        PSI_TEST_FILE = "psi_variable_Retina___Eye_psi_MERGED.pkl",
        mtsplice_BCE = 1,
        mode = "mtsplice", # mode: or "3p", "5p", "intronOnly", "intronexon", "mtsplice"
        train_mode = "train",
        eval_weights = "exprmnt_2025_08_17__02_17_03",
        run_num = 5,
        val_check_interval = 1.0,
        
        ##### --- pretrained weights ---
        mtsplice_weights = "exprmnt_2025_10_15__00_49_37", # all data weighted CL, 10 aug
        # mtsplice_weights = "exprmnt_2025_09_23__00_38_41", # ASCOT weighted CL, 10 aug
        # mtsplice_weights = "exprmnt_2025_07_30__13_10_26", #2 aug intronexon
        # mtsplice_weights = "exprmnt_2025_08_16__22_30_50", #2 aug intron
        # mtsplice_weights = "exprmnt_2025_08_16__20_42_52", #10 aug intronexon
        # mtsplice_weights = "exprmnt_2025_08_23__20_30_33", #10 aug intron

        ####### ntv2 weights for 5', 3' and exon ##########
        weight_3p = "exprmnt_2025_10_20__00_59_06",
        weight_5p = "exprmnt_2025_10_20__01_00_10",
        weight_exon = "exprmnt_2025_10_20__01_00_54",


        ####### resnet weights for 5', 3' and exon ##########

        # 10 aug, all data weighted CL
        # weight_5p = "exprmnt_2025_10_15__00_47_32",
        # weight_3p = "exprmnt_2025_10_15__00_46_22",
        # weight_exon = "exprmnt_2025_10_15__00_45_15",

        # 2 aug
        # weight_5p = "exprmnt_2025_06_01__21_15_08",
        # weight_3p = "exprmnt_2025_06_01__21_16_19",
        # weight_exon = "exprmnt_2025_06_08__21_34_21",

        # 10 aug
        # weight_5p = "exprmnt_2025_08_23__21_14_26",
        # weight_3p = "exprmnt_2025_07_08__20_39_38",
        # weight_exon = "exprmnt_2025_08_23__21_20_33",
       ################### --- psi specific parameters ################### 
        )
    return cfg




# --------------------------------------------------------------------
# 4. MAIN ENTRY
# --------------------------------------------------------------------
def main():
    cfg = get_experiment_config()
    paths = setup_paths()
    create_readme(paths["data_dir"], cfg["readme_comment"])
    submit_job(cfg, paths)


if __name__ == "__main__":
    main()
