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
        slurm_file_name = 'CL_Chrom_Split',
        task = "introns_cl", # "psi_regression_task" or "introns_cl"
        maxpooling = True,
        max_epochs = 25,
        optimizer = "adam",
        readme_comment = "CL chromosome data split\n",
        wandb_logger_NOTES = "CL chromosome data split",
        new_project_wandb = 1, # only 1 for psi
        fivep_ovrhang = 300,
        threep_ovrhang = 300,
        learning_rate =  1e-3, # Best Psi, CL: 1e-3
        global_batch_size = 2048, # Best Psi: 1024, CL: 2048
        accumulate_grad_batches = 1, # Best Psi, CL: 1
        
        ##### --- machine configuration
        gpu_num = 1,
        hour = 3,
        memory = 100,        # GB
        nthred = 8,          # CPUs

        ##### --- embedder specific
        embedder = "mtsplice",
        tokenizer = "onehot_tokenizer",
        tokenizer_seq_len = 400,
        
        ################### --- CL specific parameters ################### 
        loss_name = "supcon",
        fixed_species = False,
        n_augmentations = 10,
        temperature     = 0.2, # Best Psi, CL: 1
    
        # TRAIN_FILE="train_merged_filtered_min30Views.pkl",
        # VAL_FILE="val_merged_filtered_min30Views.pkl",    
        # TEST_FILE="test_merged_filtered_min30Views.pkl",
        TRAIN_FILE="chromSplit/train_chr_merged_filtered_min30Views.pkl",
        VAL_FILE="chromSplit/val_chr_merged_filtered_min30Views.pkl",
        TEST_FILE="chromSplit/test_chr_merged_filtered_min30Views.pkl",
        ################### --- CL specific parameters ################### 

        ################### --- psi specific parameters ################### 
        freeze_encoder = False,
        warm_start = True,
        ascot = True, # if ASCOT dataset the true, if Tabula Sapiens false     
        psi_loss_name = "MTSpliceBCELoss",
        PSI_TEST_FILE = "psi_variable_Retina___Eye_psi_MERGED.pkl", # "psi_test_Retina___Eye_psi_MERGED.pkl",
        # PSI_TEST_FILE = "psi_test_Retina___Eye_psi_MERGED.pkl", # "psi_test_Retina___Eye_psi_MERGED.pkl",
        mtsplice_BCE = 1,
        mode = "mtsplice", # mode: or "3p", "5p", "intronOnly", "intronexon", "mtsplice"
        train_mode = "train",
        eval_weights = "exprmnt_2025_08_17__02_17_03",
        run_num = 1,
        val_check_interval = 1.0,
        dropout_rate = 0.1, # Best ASCOT_Psi: 0.1, TS: 0.4
        
        ##### --- pretrained weights ---
        ####### mtsplice weights ##########
        
        # after CL hyperparameter sweep
             # intron+exon
        # mtsplice_weights = "exprmnt_2025_11_01__12_08_52",  # CL swept, 300bp, no exon pad, 10 aug
        # mtsplice_weights = "exprmnt_2025_11_01__12_16_52",  # CL swept, 300bp, no exon pad, 5 aug
        # mtsplice_weights = "exprmnt_2025_11_01__13_07_04",  # EMPRAICL_afterSweep_aug10_200bp_2025_11_01__13_07_04
        # mtsplice_weights = "exprmnt_2025_11_01__13_07_53",  # EMPRAICL_afterSweep_aug5_200bp_2025_11_01__13_07_53
            # intron only
        # mtsplice_weights = "exprmnt_2025_11_01__22_56_28",  # EMPRAICL_afterSweep_aug10_300bp_INTRON_SupCon_2025_11_01__22_56_28
        # mtsplice_weights = "exprmnt_2025_11_01__22_58_09",  # EMPRAICL_afterSweep_aug5_300bp_INTRON_SupCon_2025_11_01__22_58_09
        # mtsplice_weights = "exprmnt_2025_11_01__22_51_48",  # EMPRAICL_afterSweep_aug10_200bp_INTRON_SupCon_2025_11_01__22_51_48
        # mtsplice_weights = "exprmnt_2025_11_01__22_51_25",  # EMPRAICL_afterSweep_aug5_200bp_INTRON_SupCon_2025_11_01__22_51_25

        # after recomb review
             # intron+exon
        # mtsplice_weights = "exprmnt_2026_01_27__23_16_47",  # CL swept, 300bp, no exon pad, 10 aug, check, 15 epochs
        # mtsplice_weights = "exprmnt_2026_01_28__00_43_17",  # CL swept, 300bp, no exon pad, 10 aug, check, 25 epochs
        # mtsplice_weights = "exprmnt_2026_01_29__13_00_53",  # CL swept, 300bp, no exon pad, 10 aug, CNN_6L
        # mtsplice_weights = "exprmnt_2026_01_29__13_01_25",  # CL swept, 300bp, no exon pad, 10 aug, CNN_4L
        # mtsplice_weights = "exprmnt_2026_01_29__13_01_48",  # CL swept, 300bp, no exon pad, 10 aug, CNN_2L
        mtsplice_weights = "exprmnt_2026_06_13__00_11_47",
       
        ####### ntv2 weights for 5', 3' and exon ##########
        weight_3p = "exprmnt_2025_10_20__00_59_06",
        weight_5p = "exprmnt_2025_10_20__01_00_10",
        weight_exon = "exprmnt_2025_10_20__01_00_54",

       ################### --- psi specific parameters ################### 
        )
    return cfg




# --------------------------------------------------------------------
# 4. MAIN ENTRY
# --------------------------------------------------------------------
def main():
    cfg = get_experiment_config()
    if not cfg["ascot"]:
        cfg["dropout_rate"] = 0.4
    else:
        cfg["dropout_rate"] = 0.1

    paths = setup_paths()
    create_readme(paths["data_dir"], cfg["readme_comment"])
    submit_job(cfg, paths)


if __name__ == "__main__":
    main()
