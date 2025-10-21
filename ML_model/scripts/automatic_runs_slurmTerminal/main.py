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
        which_part = "pretraining", # "pretraining" or "finetuning"
        slurm_file_name = 'Psi_newruntry',
        
        ##### --- machine configuration
        gpu_num = 1,
        hour = 1,
        memory = 100,        # GB
        nthred = 8,          # CPUs

        ##### --- embedder specific
        embedder = "ntv2",
        tokenizer = "hf_tokenizer",
        tokenizer_seq_len = 201,
        # embedder = "resnet",
        # tokenizer = "custom_tokenizer",
        # tokenizer_seq_len = 201,
        # embedder = "mtsplice",
        # tokenizer = "onehot_tokenizer",
        # tokenizer_seq_len = 400,
        
        
        ##### --- CL specific parameters
        task = "introns_cl",
        loss_name = "supcon",
        fixed_species = False,
        maxpooling = True,
        n_augmentations = 2,
        # TRAIN_FILE="train_3primeIntron_filtered_min30views.pkl",
        # VAL_FILE="val_3primeIntron_filtered.pkl",
        # TEST_FILE="test_3primeIntron_filtered.pkl",

        # TRAIN_FILE="train_5primeIntron_filtered.pkl",
        # VAL_FILE="val_5primeIntron_filtered.pkl",
        # TEST_FILE="test_5primeIntron_filtered.pkl",

        TRAIN_FILE="train_ExonSeq_filtered.pkl",
        VAL_FILE="val_ExonSeq_filtered.pkl",
        TEST_FILE="test_ExonSeq_filtered.pkl",

        # TRAIN_FILE="train_merged_filtered_min30Views.pkl",
        # VAL_FILE="val_merged_filtered_min30Views.pkl",    
        # TEST_FILE="test_merged_filtered_min30Views.pkl",

        # TRAIN_FILE="ASCOT_data/train_ASCOT_merged_filtered_min30Views.pkl",
        # VAL_FILE="ASCOT_data/val_ASCOT_merged_filtered_min30Views.pkl",
        # TEST_FILE="ASCOT_data/test_ASCOT_merged_filtered_min30Views.pkl",

        # TRAIN_FILE="ASCOT_data/train_ASCOT_3primeIntron_filtered.pkl",
        # VAL_FILE="ASCOT_data/val_ASCOT_3primeIntron_filtered.pkl",
        # TEST_FILE="ASCOT_data/test_ASCOT_3primeIntron_filtered.pkl",

        # TRAIN_FILE="ASCOT_data/train_ASCOT_5primeIntron_filtered.pkl",
        # VAL_FILE="ASCOT_data/val_ASCOT_5primeIntron_filtered.pkl",
        # TEST_FILE="ASCOT_data/test_ASCOT_5primeIntron_filtered.pkl",

        # TRAIN_FILE="ASCOT_data/train_ASCOT_ExonSeq_filtered.pkl",
        # VAL_FILE="ASCOT_data/val_ASCOT_ExonSeq_filtered.pkl",
        # TEST_FILE="ASCOT_data/test_ASCOT_ExonSeq_filtered.pkl",
        ##### --- CL specific parameters --- #####

        ##### --- psi specific parameters --- #####
        freeze_encoder = False,
        warm_start = True,
        psi_loss_name = "MTSpliceBCELoss",
        PSI_TEST_FILE = "psi_variable_Retina___Eye_psi_MERGED.pkl",
        mtsplice_BCE = 1,
        mode = "intronexon",
        train_mode = "train",
        eval_weights = "exprmnt_2025_08_17__02_17_03",
        run_num = 10,
        val_check_interval = 1.0,
        global_batch_size = 256,
        max_epochs = 20,
        optimizer = "sgd",
        learning_rate =  1e-3,
        
        ##### --- pretrained weights ---
        ####### mtsplice weights ##########
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
        ##### --- psi specific parameters --- #####

        readme_comment = "new trial\n",
        wandb_logger_NOTES = "new trial",
        new_project_wandb = 1,
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