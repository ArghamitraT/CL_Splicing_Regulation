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
# 👇 Combined folder name
FINAL_DIR_NAME="combined_runs"

# 👇 Input full experiment names (space-separated)
# Example:
# EMPRAIPsi_200bp_MTCLSwept_10Aug_noExonPad_2025_11_01__18_59_24
# EMPRAIPsi_200bp_MTCLSwept_5Aug_noExonPad_2025_11_01__18_57_08
read -p "Enter experiment names (space-separated): " -a EXP_NAMES

# ===============================
# DERIVE exprmnt_* FOLDER NAMES
# ===============================
EXPERIMENTS=()
for name in "${EXP_NAMES[@]}"; do
    # Extract date/time part (pattern: 2025_11_01__18_59_24)
    datetime=$(echo "$name" | grep -oE '[0-9]{4}_[0-9]{2}_[0-9]{2}__[0-9]{2}_[0-9]{2}_[0-9]{2}')
    if [ -n "$datetime" ]; then
        EXPERIMENTS+=("exprmnt_${datetime}")
    else
        echo "⚠️  Skipping invalid name (no datetime found): $name"
    fi
done

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
