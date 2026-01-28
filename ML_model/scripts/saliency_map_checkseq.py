from pathlib import Path
from omegaconf import OmegaConf
import os, sys, time, json, logging, numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import hydra
import seaborn as sns
import os, sys, random, logging, torch
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
import hydra
import time

timestamp = time.strftime("_%Y_%m_%d__%H_%M_%S")

# ---------------- root discovery ----------------
from pathlib import Path
def find_contrastive_root(start: Path = Path(__file__)) -> Path:
    for parent in start.resolve().parents:
        if parent.name == "Contrastive_Learning":
            return parent
    raise RuntimeError("Could not find 'Contrastive_Learning' directory.")

root_path = str(find_contrastive_root())
os.environ["CONTRASTIVE_ROOT"] = root_path
print(f"CONTRASTIVE_ROOT set to: {root_path}")

# ---------------- imports from your repo ----------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.config import print_config
from src.datasets.auxiliary_jobs import PSIRegressionDataModule
from src.utils.encoder_init import initialize_encoders_and_model   # <-- we will only take the encoder features
from src.datasets.utility import get_windows_with_padding

# ---------------- user overrides ----------------
ANCHOR_EXON_ID   = os.environ.get("ANCHOR_EXON_ID",  "")
NEGATIVE_EXON_ID = os.environ.get("NEGATIVE_EXON_ID","")
TISSUE_NAME      = os.environ.get("TISSUE_NAME",     "lung")  # not strictly needed here, but kept for metadata
SIM_TYPE         = os.environ.get("SIM_TYPE", "cosine")       # "cosine" or "dot"
RESULT_DIR       = os.environ.get("RESULT_DIR", "exprmnt_2025_11_01__22_56_28") # "exprmnt_2025_10_25__15_31_32"
comment = "EMPRAICL_afterSweep_aug10_300bp_INTRON_SupCon_2025_11_01__22_56_28"

# timestamp = time.strftime("_%Y_%m_%d__%H_%M_%S")
# main_dir = f"{root_path}/code/ML_model"
# os.makedirs(f"{main_dir}/figures", exist_ok=True)
# os.makedirs(f"{main_dir}/arrays",  exist_ok=True)

# ---------------- logging ----------------
def setup_logging(save_dir: Path, filename: str = "contrastive_saliency.log"):
    save_dir.mkdir(parents=True, exist_ok=True)
    p = save_dir / filename
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    fh = logging.FileHandler(p, mode="w"); fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt); ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh); logger.addHandler(ch)
    logging.info(f"Logging to {p}")


import os
import numpy as np
from collections import defaultdict



def process_seq(entry_A):
    
    entry_A["5p"] = entry_A["5p"][-200:]
    entry_A["3p"] = entry_A["3p"][:200]

    start = entry_A["exon"].get("start", "")
    end = entry_A["exon"].get("end", "")
            
    full_seq =  entry_A["5p"] + start+end + entry_A["3p"]

    windows = get_windows_with_padding(300, 300, 100, 100, full_seq, overhang = (200, 200))

    return windows['acceptor']+windows['donor']




import pickle

main_dir = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/extra/contrast_saliency_random/f__2025_11_19__13_37_51/figures"
# ---------------- HYDRA MAIN ----------------
@hydra.main(version_base=None, config_path="../configs", config_name="psi_regression.yaml")
def main(config: OmegaConf):
    # logging

    # out_dir = f"{root_path}/data/extra/contrast_saliency_random/f_{timestamp}"
    # out_dir = Path(out_dir)
    # out_dir.mkdir(parents=True, exist_ok=True)
    # setup_logging(out_dir)

    # resolvers
    OmegaConf.register_new_resolver('eval', eval)
    OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
    OmegaConf.register_new_resolver('min', lambda x, y: min([x, y]))
    OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
    OmegaConf.register_new_resolver('optimal_workers', lambda: min(os.cpu_count() // max(1, torch.cuda.device_count()), 8))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- configure PSI dataset (you were already doing this) ---
    OmegaConf.set_struct(config, False)
    # Point to your PSI pkl for the chosen tissue
    # (You had Retina/Eye here; keep or override via Hydra/ENV as you prefer)
    config.dataset.test_files.intronexon = f"{root_path}/data/final_data/ASCOT_finetuning/psi_variable_Retina___Eye_psi_MERGED.pkl"
    config.dataset.fivep_ovrhang = 200
    config.dataset.threep_ovrhang = 200
    config.aux_models.mtsplice_weights = RESULT_DIR
    config.aux_models.warm_start = True
    config.aux_models.mode = "mtsplice"
    config.task.global_batch_size = 2048
    OmegaConf.set_struct(config, True)

    print_config(config, resolve=True)
    # logging.info(f"🎯 CL_weight: {RESULT_DIR}")
    # logging.info(f"{comment}")

    # --- Data ---
    data_module = PSIRegressionDataModule(config)
    data_module.setup()
    # logging.info(f"Dataset size (test): {len(data_module.test_set)}; keys available: {len(getattr(data_module.test_set,'data',{}))}")
    
    high_sample = ['GT_62125', 'GT_34667', 'GT_05391', 'GT_24765', 'GT_30881', 'GT_53818', 'GT_42813', 'GT_56990', 'GT_25156', 'GT_77786', 'GT_43914', 'GT_28682', 'GT_04544', 'GT_11538', 'GT_70831', 'GT_14195', 'GT_33825', 'GT_73433', 'GT_19230', 'GT_02570']
    low_sample  = ['GT_09591', 'GT_58515', 'GT_53963', 'GT_71001', 'GT_45119', 'GT_20109', 'GT_21527', 'GT_06314', 'GT_19466', 'GT_25036', 'GT_79749', 'GT_29902', 'GT_38954', 'GT_43067', 'GT_68350', 'GT_05383', 'GT_53279', 'GT_76431', 'GT_27762', 'GT_57800']

    # Stores for logo-making
    high_left = []
    low_left = []
    high_right = []
    low_right = []
    high_fullSeq = []
    low_fullSeq = []

    left_start, left_end = 296, 302
    right_start, right_end = 98, 104

    for anchor_id in high_sample:
        if anchor_id not in data_module.test_set.data:
            # logging.warning(f"⚠️ Anchor exon {anchor_id} not found in dataset. Skipping.")
            continue
        
        entry_A = data_module.test_set.data.get(anchor_id, None)
        fullseqA = process_seq(entry_A)
        high_fullSeq.append(fullseqA)

        # extract windows
        left_win  = fullseqA[left_start:left_end]
        right_win = fullseqA[-400:][right_start:right_end]
        high_left.append(left_win)
        high_right.append(right_win)
        print(f"Processed POS {anchor_id}: left_win='{left_win}', right_win='{right_win}'")
        
        # if len(left_win)  == (left_end - left_start):
        #     high_left.append(left_win)
        # if len(right_win) == (right_end - right_start):
        #     high_right.append(right_win)

        # fullSeqA = process_seq(entry_A)
        # high_fullSeq.append(fullSeqA)

        # # extract windows
        # left_win  = fullSeqA[left_start:left_end]
        # right_win = fullSeqA[right_start:right_end]
        
        # if len(left_win)  == (left_end - left_start):
        #     high_left.append(left_win)
        # if len(right_win) == (right_end - right_start):
        #     high_right.append(right_win)

        
           
    for negative_id in low_sample:
        if negative_id not in data_module.test_set.data:
            # logging.warning(f"⚠️ Negative exon {negative_id} not found in dataset. Skipping.")
            continue
        entry_N = data_module.test_set.data.get(negative_id, None)
        fullSeqN = process_seq(entry_N)
        low_fullSeq.append(fullSeqN)

        left_win  = fullSeqN[left_start:left_end]
        right_win = fullSeqN[-400:][right_start:right_end]
        low_left.append(left_win)
        low_right.append(right_win)
        print(f"Processed NEG {negative_id}: left_win='{left_win}', right_win='{right_win}'")
        

        # left_win  = fullSeqN[left_start:left_end]
        # right_win = fullSeqN[right_start:right_end]

        # if len(left_win) == (left_end - left_start):
        #     low_left.append(left_win)
        # if len(right_win) == (right_end - right_start):
        #     low_right.append(right_win)

    save_path = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/extra/contrast_saliency_random/f__2025_11_19__13_37_51/pickles/motif_windows.pkl"
    
    left_start, left_end = 197, 203
    right_start, right_end = 399, 405

    motif_data = {
    "HIGH_fullSeq": high_fullSeq,
    "LOW_fullSeq":  low_fullSeq,
    "HIGH_left":  high_left,
    "LOW_left":   low_left,
    "HIGH_right": high_right,
    "LOW_right":  low_right,
    "left_range":  (left_start, left_end),
    "right_range": (right_start, right_end),
}
    with open(save_path, "wb") as f:
        pickle.dump(motif_data, f)

    print(f"Saved motif windows to:\n{save_path}")
    print("Counts:")
    print("HIGH left:", len(high_left))
    print("LOW left:",  len(low_left))
    print("HIGH right:", len(high_right))
    print("LOW right:",  len(low_right))

    
    # print("Collected windows:")
    # print("HIGH left:", len(high_left))
    # print("LOW left:",  len(low_left))
    # print("HIGH right:", len(high_right))
    # print("LOW right:",  len(low_right))


    # LEFT splice site logos
    # make_logo(high_left, "HIGH_Left_splice_site_197_201","HIGH Motif (Left splice site: 197–201)")
    # make_logo(low_left,  "LOW_Left_splice_site_197_201","LOW Motif (Left splice site: 197–201)")

    # # RIGHT splice site logos
    # make_logo(high_right, "HIGH_Right_splice_site_398_403","HIGH Motif (Right splice site: 398–403)")
    # make_logo(low_right,  "LOW_Right_splice_site_398_403","LOW Motif (Right splice site: 398–403)")






    print()

    # logging.info("✅ Completed all contrastive saliency combinations.")



if __name__ == "__main__":
    main()