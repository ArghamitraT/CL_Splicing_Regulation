import pandas as pd
import pickle

# # ---------------------------------------------------------
# # Paths — EDIT THESE
# # ---------------------------------------------------------
# HIGH_CSV = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/variable_cassette_exons_with_binary_labels_HIGH_TissueBinPsi.csv"
# LOW_CSV  = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/ASCOT/variable_cassette_exons_with_binary_labels_LOW_TissueBinPsi.csv"
# PKL_PATH = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data/ASCOT_finetuning/psi_variable_Retina___Eye_psi_MERGED.pkl"

# # ---------------------------------------------------------
# # Load CSVs
# # ---------------------------------------------------------
# df_high = pd.read_csv(HIGH_CSV)
# df_low  = pd.read_csv(LOW_CSV)

# high_ids = df_high["exon_id"].astype(str).tolist()
# low_ids  = df_low["exon_id"].astype(str).tolist()

# print("Loaded HIGH:", len(high_ids))
# print("Loaded LOW:", len(low_ids))

# # ---------------------------------------------------------
# # Load PKL
# # ---------------------------------------------------------
# with open(PKL_PATH, "rb") as f:
#     seq_dict = pickle.load(f)

# Example expected structure:
# seq_dict["GT_00543"] = {
#      "5p": "....",
#      "3p": "....",
#      "exon": { "start": ..., "end": ... }
# }

# ---------------------------------------------------------
# Helper: check GT–AG rule
# ---------------------------------------------------------
def good_splice(entry):
    seq_5p = entry["5p"].upper()
    seq_3p = entry["3p"].upper()
    return seq_5p.endswith("AG") and seq_3p.startswith("GT")

# # ---------------------------------------------------------
# # Apply filter
# # ---------------------------------------------------------
# high_good = []
# for exon_id in high_ids:
#     if exon_id in seq_dict and good_splice(seq_dict[exon_id]):
#         high_good.append(exon_id)

# low_good = []
# for exon_id in low_ids:
#     if exon_id in seq_dict and good_splice(seq_dict[exon_id]):
#         low_good.append(exon_id)

# # ---------------------------------------------------------
# # Percentages
# # ---------------------------------------------------------
# pct_high = 100 * len(high_good) / len(high_ids)
# pct_low  = 100 * len(low_good)  / len(low_ids)

# # ---------------------------------------------------------
# # Results
# # ---------------------------------------------------------
# print("\n=== GT–AG FILTER RESULTS ===")
# print(f"High GT–AG: {len(high_good)} / {len(high_ids)}  ({pct_high:.2f}%)")
# print(f"Low  GT–AG: {len(low_good)} / {len(low_ids)}   ({pct_low:.2f}%)")

# print("\nHigh Good List:")
# print(high_good)

# print("\nLow Good List:")
# print(low_good)







import csv
import pickle

# -------------------------------------------------------
# Load CSV
# -------------------------------------------------------
csv_path = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/extra/common_high_low_exons.csv"

High_list= ['GT_62125', 'GT_34667', 'GT_05391', 'GT_24765', 'GT_30881', 'GT_53818', 'GT_42813', 'GT_56990', 'GT_25156', 'GT_77786', 'GT_43914', 'GT_28682', 'GT_04544', 'GT_11538', 'GT_70831', 'GT_14195', 'GT_33825', 'GT_73433', 'GT_19230', 'GT_53606', 'GT_33365', 'GT_14004', 'GT_20363', 'GT_44455', 'GT_12895', 'GT_13999', 'GT_18172', 'GT_57920', 'GT_16155', 'GT_41494', 'GT_12206', 'GT_68205', 'GT_42811', 'GT_46777', 'GT_09758', 'GT_74995', 'GT_30459', 'GT_50502', 'GT_61757', 'GT_73662', 'GT_29351', 'GT_11218', 'GT_34859', 'GT_24528', 'GT_22981', 'GT_05530', 'GT_32397', 'GT_51171', 'GT_43747', 'GT_33722', 'GT_67752', 'GT_10563', 'GT_06975', 'GT_55879', 'GT_32878', 'GT_01635', 'GT_04102', 'GT_22273', 'GT_22343', 'GT_28852', 'GT_45206', 'GT_77994', 'GT_24365', 'GT_06007', 'GT_01962', 'GT_22951', 'GT_75936', 'GT_59788', 'GT_02625', 'GT_64141', 'GT_32874', 'GT_36765', 'GT_08106', 'GT_53813', 'GT_67544', 'GT_32877', 'GT_04460', 'GT_67750', 'GT_64438', 'GT_38139', 'GT_31481', 'GT_32510', 'GT_31830', 'GT_43764', 'GT_70858', 'GT_37754', 'GT_50445', 'GT_76886', 'GT_20117', 'GT_47707', 'GT_29346', 'GT_68070', 'GT_16617', 'GT_01293', 'GT_74994', 'GT_64427', 'GT_24980', 'GT_66167', 'GT_12533', 'GT_03764', 'GT_22963', 'GT_03979', 'GT_32799', 'GT_20088', 'GT_46554', 'GT_69599', 'GT_33724', 'GT_30880', 'GT_16400', 'GT_64082', 'GT_21931', 'GT_63230', 'GT_64257', 'GT_22601', 'GT_66306', 'GT_02613', 'GT_39019']
Low_list= ['GT_09591', 'GT_58515', 'GT_53963', 'GT_71001', 'GT_45119', 'GT_20109', 'GT_21527', 'GT_06314', 'GT_19466', 'GT_25036', 'GT_79749', 'GT_29902', 'GT_38954', 'GT_43067', 'GT_68350', 'GT_05383', 'GT_53279', 'GT_76431', 'GT_41272', 'GT_33389', 'GT_29684', 'GT_22966', 'GT_60723', 'GT_71753', 'GT_37568', 'GT_25576', 'GT_74539', 'GT_00781', 'GT_12056', 'GT_25227', 'GT_61414', 'GT_19431', 'GT_66920', 'GT_75419', 'GT_18058', 'GT_63124', 'GT_04103', 'GT_65296', 'GT_11537', 'GT_26196', 'GT_50384', 'GT_01118', 'GT_14995', 'GT_12712', 'GT_39021', 'GT_45848', 'GT_00831', 'GT_15371', 'GT_16724', 'GT_05834', 'GT_04984', 'GT_48486', 'GT_43076', 'GT_73080', 'GT_16180', 'GT_62126', 'GT_48489', 'GT_72612', 'GT_02620', 'GT_50222', 'GT_74791', 'GT_26448', 'GT_62370', 'GT_39421', 'GT_75939', 'GT_64977', 'GT_67588', 'GT_19460', 'GT_36991', 'GT_27898', 'GT_41023', 'GT_50453', 'GT_39599', 'GT_51159', 'GT_64255', 'GT_02949', 'GT_64889', 'GT_50375', 'GT_01946', 'GT_72317', 'GT_00870', 'GT_04642', 'GT_22070', 'GT_20089', 'GT_39026', 'GT_55024', 'GT_55948', 'GT_28636', 'GT_58206', 'GT_21347', 'GT_70344', 'GT_04110', 'GT_48477', 'GT_42809', 'GT_16327', 'GT_48479', 'GT_25878', 'GT_28628', 'GT_20796', 'GT_08428', 'GT_47774', 'GT_58714', 'GT_06899', 'GT_30467', 'GT_23178', 'GT_27065', 'GT_05031', 'GT_31890', 'GT_01139', 'GT_25578', 'GT_42399', 'GT_67103', 'GT_19212', 'GT_24406', 'GT_23713', 'GT_04449', 'GT_36687', 'GT_34211', 'GT_37764', 'GT_25141', 'GT_43070', 'GT_34862', 'GT_40615', 'GT_50578', 'GT_23158', 'GT_47533', 'GT_22602', 'GT_48167', 'GT_61804', 'GT_03762', 'GT_47668', 'GT_54950', 'GT_38501', 'GT_64433', 'GT_34705', 'GT_52782', 'GT_69827', 'GT_63598', 'GT_52645', 'GT_67916', 'GT_70581', 'GT_16184', 'GT_75934', 'GT_19584']



pairs = []
with open(csv_path) as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        high, low = row
        pairs.append((high.strip(), low.strip()))



# Flatten CSV entries
csv_high = [h for (h, _) in pairs]
csv_low  = [l for (_, l) in pairs]

# Convert your existing lists to sets for fast lookup
High_set = set(High_list)
Low_set  = set(Low_list)

# Find which IDs from CSV are NOT in your provided lists
missing_high = [h for h in csv_high if h not in High_set]
missing_low  = [l for l in csv_low  if l not in Low_set]

# Combine both sides
missing_all = missing_high + missing_low

print("Missing HIGH entries:", missing_high)
print("Missing LOW entries :", missing_low)
print("Total missing:", len(missing_all))



# # -------------------------------------------------------
# # Load CSV
# # -------------------------------------------------------
# csv_path = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/extra/common_high_low_exons.csv"


# pairs = []
# with open(csv_path) as f:
#     reader = csv.reader(f)
#     next(reader)  # skip header
#     for row in reader:
#         high, low = row
#         pairs.append((high.strip(), low.strip()))

# # -------------------------------------------------------
# # Load PKL with sequence entries
# # -------------------------------------------------------
# pkl_path = "/gpfs/commons/home/atalukder/Contrastive_Learning/data/final_data_old/ASCOT_finetuning/psi_variable_Retina___Eye_psi_MERGED.pkl"
# with open(pkl_path, "rb") as f:
#     seq_dict = pickle.load(f)

# # seq_dict should contain records like:
# # seq_dict["GT_12345"] = {
# #     "5p": "...",
# #     "3p": "...",
# #     "exon": {"start":..., "end":...}
# # }

# # -------------------------------------------------------
# # Helpers
# # -------------------------------------------------------
# def is_gt_ag(entry):
#     """Return True if 5p ends with GT AND 3p begins with AG."""
#     seq_5p = entry["5p"][-200:].upper()
#     seq_3p = entry["3p"][:200].upper()
#     return seq_5p.endswith("AG") and seq_3p.startswith("GT")

# # -------------------------------------------------------
# # Process high and low lists
# # -------------------------------------------------------
# high_good = []
# low_good = []

# for high_id, low_id in pairs:
#     if high_id in seq_dict:
#         if is_gt_ag(seq_dict[high_id]):
#             high_good.append(high_id)

#     if low_id in seq_dict:
#         if is_gt_ag(seq_dict[low_id]):
#             low_good.append(low_id)

# # -------------------------------------------------------
# # Compute percentages
# # -------------------------------------------------------
# pct_high = 100 * len(high_good) / len(pairs)
# pct_low  = 100 * len(low_good)  / len(pairs)

# # -------------------------------------------------------
# # Final output
# # -------------------------------------------------------
# print("=== RESULTS ===")
# print("High exons satisfying GT–AG rule:", len(high_good), "/", len(pairs),
#       f"({pct_high:.2f}%)")
# print("Low exons satisfying GT–AG rule :", len(low_good), "/", len(pairs),
#       f"({pct_low:.2f}%)")

# print("\nHigh list:", high_good)
# print("Low list :", low_good)



