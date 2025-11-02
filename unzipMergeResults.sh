#!/bin/bash
set -e

# ===============================
# DYNAMIC PATH DETECTION
# ===============================
BASE_PATH=$(pwd | sed -E 's|(.*Contrastive_Learning).*|\1|')
RESULTS_DIR="${BASE_PATH}/files/results"

# ===============================
# INPUT SECTION
# ===============================
# 👇 Name of the zip file (without full path)
ZIP_NAME="combined_runs.zip"   # <-- change this to your actual zip name
ZIP_PATH="${RESULTS_DIR}/${ZIP_NAME}"

# ===============================
# VALIDATE INPUT
# ===============================
if [ ! -f "$ZIP_PATH" ]; then
    echo "❌ Zip file not found: $ZIP_PATH"
    exit 1
fi

# Temporary extraction folder (same name as zip without .zip)
EXTRACT_DIR="${RESULTS_DIR}/${ZIP_NAME%.zip}"

echo "📦 Zip file:         $ZIP_PATH"
echo "📂 Extract to:       $EXTRACT_DIR"
echo "📁 Target directory: $RESULTS_DIR"
echo

# ===============================
# UNZIP INTO TEMPORARY FOLDER
# ===============================
echo "🗜️  Extracting zip file..."
unzip -q "$ZIP_PATH" -d "$EXTRACT_DIR"

# ===============================
# MOVE CONTENTS TO MAIN FOLDER
# ===============================
echo "🚚 Moving extracted experiment folders into main results directory..."

shopt -s dotglob  # include hidden files

# If the zip contains a single parent folder (like combined_runs/exprmnt_*),
# move from one level deeper.
if [ "$(find "$EXTRACT_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 1 ]; then
    INNER_DIR=$(find "$EXTRACT_DIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)
    echo "📁 Detected inner directory: $INNER_DIR"
    mv "$INNER_DIR"/* "$RESULTS_DIR"/ 2>/dev/null || echo "⚠️  Nothing to move from inner directory."
else
    mv "$EXTRACT_DIR"/* "$RESULTS_DIR"/ 2>/dev/null || echo "⚠️  Nothing to move."
fi

shopt -u dotglob

# ===============================
# CLEANUP
# ===============================
echo "🧹 Cleaning up..."
rm -rf "$EXTRACT_DIR"
rm -f "$ZIP_PATH"

echo
echo "✅ Done!"
echo "🗂️  All extracted experiment folders moved to: $RESULTS_DIR"
echo "🧽 Removed temporary folder and original zip file."

