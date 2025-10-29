#!/bin/bash
set -e

# ===============================
# DYNAMIC PATH DETECTION
# ===============================
BASE_PATH=$(pwd | sed -E 's|(.*Contrastive_Learning).*|\1|')
SERVER_RESULT_BASE="${BASE_PATH}/files/results"

# ===============================
# USER INPUT SECTION
# ===============================
# 👇 Change this to your desired combined folder name
FINAL_DIR_NAME="combined_runs"

# 👇 List your experiment folders here (space separated)
# EXPERIMENTS=(
# exprmnt_2025_10_20__15_18_32
# exprmnt_2025_10_20__15_11_51
# exprmnt_2025_10_19__16_56_25
# exprmnt_2025_10_15__18_14_13
# )

# EXPERIMENTS=(
# exprmnt_2025_10_22__23_40_20 # MTsplice new model
# exprmnt_2025_10_22__23_35_53 # MTsplice new model, CL, MSE, confident species
# exprmnt_2025_10_22__23_34_30 # MTsplice new model, CL, BCE, confident species
# )

# EXPERIMENTS=(
# exprmnt_2025_10_24__16_11_26 # EMPRAIPsi_MTSplNew_CnfdntSpcs_BCE_wtdSupCon
# exprmnt_2025_10_24__16_07_40 # EMPRAIPsi_MTSplNew_AllSpcs_BCE_wtdSupCon
# exprmnt_2025_10_24__16_05_34 # EMPRAIPsi_MTSplNew_AllSpcs_BCE
# exprmnt_2025_10_24__04_01_01 # EMPRAIPsi_MTSpliceNewmodel_300bpIntron
# )

# EXPERIMENTS=(
# exprmnt_2025_10_25__22_41_13 # EMPRAIPsi_MTSplNew_AllSpcs_BCE_noASCOTTestinTrain
# exprmnt_2025_10_25__22_38_22 # EMPRAIPsi_MTSplNew_AllSpcs_BCE_wtdSupCon_noASCOTTestinTrain
# )
EXPERIMENTS=(
 exprmnt_2025_10_28__20_12_11 # intron ofset 300 bp like MTsplice, CL wtdSupcon, MTSplice hyperparameters
 exprmnt_2025_10_28__20_12_58 # intron ofset 300 bp like MTsplice, CL normal Supcon, MTSplice hyperparameters
 exprmnt_2025_10_28__20_28_29 # intron ofset 200 bp like MTsplice, CL normal Supcon, MTSplice hyperparameters
 exprmnt_2025_10_28__20_30_30 # intron ofset 200 bp like MTsplice, CL weighted Supcon, MTSplice hyperparameters
)

# ===============================
# PATH SETUP
# ===============================
DEST_DIR="${SERVER_RESULT_BASE}/${FINAL_DIR_NAME}"

# Create destination folder if missing
if [ ! -d "$DEST_DIR" ]; then
    echo "📁 Creating destination directory: $DEST_DIR"
    mkdir -p "$DEST_DIR"
fi

echo "📂 Base path:          $BASE_PATH"
echo "📁 Source directory:   $SERVER_RESULT_BASE"
echo "📦 Destination folder: $DEST_DIR"
echo

# ===============================
# COPY LOOP (exclude all .ckpt files)
# ===============================
for EXP in "${EXPERIMENTS[@]}"; do
    SRC="${SERVER_RESULT_BASE}/${EXP}"
    if [ -d "$SRC" ]; then
        echo "🚀 Copying $EXP → $DEST_DIR (excluding .ckpt)"
        rsync -av --exclude='*.ckpt' "$SRC" "$DEST_DIR/"
    else
        echo "⚠️  Skipping: $SRC not found."
    fi
done

echo
echo "✅ Done! All selected experiment folders copied (no .ckpt files)."
# ===============================
# ZIP THE FINAL FOLDER
# ===============================
cd "$SERVER_RESULT_BASE" || { echo "❌ Failed to enter $SERVER_RESULT_BASE"; exit 1; }

ZIP_NAME="${FINAL_DIR_NAME}.zip"

# Remove old zip if exists
if [ -f "$ZIP_NAME" ]; then
    echo "🧹 Removing old archive: $ZIP_NAME"
    rm "$ZIP_NAME"
fi

echo "🗜️  Zipping $FINAL_DIR_NAME → $ZIP_NAME ..."
zip -r "$ZIP_NAME" "$FINAL_DIR_NAME" > /dev/null

echo
echo "🎉 Done! Combined folder zipped at:"
echo "   $SERVER_RESULT_BASE/$ZIP_NAME"
