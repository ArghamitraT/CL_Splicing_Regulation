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
FINAL_DIR_NAME="combined_runs"

read -p "Enter experiment names (space-separated): " -a EXP_NAMES

# ===============================
# DERIVE exprmnt_* FOLDER NAMES
# ===============================
EXPERIMENTS=()
for name in "${EXP_NAMES[@]}"; do
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

if [ ! -d "$DEST_DIR" ]; then
    echo "📁 Creating destination directory: $DEST_DIR"
    mkdir -p "$DEST_DIR"
fi

echo "📂 Base path:          $BASE_PATH"
echo "📁 Source directory:   $SERVER_RESULT_BASE"
echo "📦 Destination folder: $DEST_DIR"
echo

# ===============================
# COPY LOOP (only test_set_evaluation)
# ===============================
for EXP in "${EXPERIMENTS[@]}"; do
    SRC="${SERVER_RESULT_BASE}/${EXP}/ensemble_evaluation_from_valdiation/test_set_evaluation"
    DEST="${DEST_DIR}/${EXP}/ensemble_evaluation_from_valdiation/"

    if [ -d "$SRC" ]; then
        echo "🚀 Copying ${SRC} → ${DEST}"
        mkdir -p "$DEST"
        rsync -av "$SRC" "$DEST"
    else
        echo "⚠️  Skipping: $SRC not found."
    fi
done

echo
echo "✅ Done! Only test_set_evaluation folders copied (structure preserved)."

# ===============================
# ZIP THE FINAL FOLDER
# ===============================
cd "$SERVER_RESULT_BASE" || { echo "❌ Failed to enter $SERVER_RESULT_BASE"; exit 1; }

ZIP_NAME="${FINAL_DIR_NAME}.zip"

if [ -f "$ZIP_NAME" ]; then
    echo "🧹 Removing old archive: $ZIP_NAME"
    rm "$ZIP_NAME"
fi

echo "🗜️  Zipping $FINAL_DIR_NAME → $ZIP_NAME ..."
zip -r "$ZIP_NAME" "$FINAL_DIR_NAME" > /dev/null

echo
echo "🎉 Done! Zipped folder created at:"
echo "   $SERVER_RESULT_BASE/$ZIP_NAME"
