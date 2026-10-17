#!/bin/bash

GDRIVE="gdrive"
GDRIVE_BASE="knowles_research"

# === Dynamically find current base path ===
# Finds the path up to the Contrastive_Learning directory
BASE_PATH=$(pwd | sed -E 's|(.*Contrastive_Learning).*|\1|')

# Define dynamic local paths
SERVER_DATA_BASE="${BASE_PATH}/data/final_data/intronExonSeq_multizAlignment_noDash/trainTestVal_data/chromSplit"

# Define GDrive paths
GDRIVE_DATA_PATH="${GDRIVE}:${GDRIVE_BASE}/data"

# === Ask for type ===
read -p "📂 Type of upload? (data/other): " TYPE
TYPE=$(echo "$TYPE" | tr '[:upper:]' '[:lower:]')

if [[ "$TYPE" == "data" ]]; then
    echo "📁 Available data items:"
    ls -1 "$SERVER_DATA_BASE"
    read -p "👉 Enter name to upload (file or folder): " ITEM_NAME
    SERVER_PATH="${SERVER_DATA_BASE}/${ITEM_NAME}"
    GDRIVE_PATH="${GDRIVE_DATA_PATH}"

elif [[ "$TYPE" == "result" ]]; then
    echo "📁 Available result folders:"
    ls -1 "$SERVER_RESULT_BASE"
    read -p "👉 Enter folder name to upload: " ITEM_NAME
    SERVER_PATH="${SERVER_RESULT_BASE}/${ITEM_NAME}"
    GDRIVE_PATH="${GDRIVE_RESULT_PATH}"

    echo ""
    read -p "🧠 Upload with or without checkpoints (.ckpt files)? (with/without): " CKPT_CHOICE
    CKPT_CHOICE=$(echo "$CKPT_CHOICE" | tr '[:upper:]' '[:lower:]')

else
    read -p "📂 Enter full local path to upload: " SERVER_PATH
    read -p "☁️  Enter full remote path (e.g. gdrive_columbia:technical_work/CntrstvLrn_FuncExon): " GDRIVE_PATH
fi

# === Confirm paths ===
echo ""
echo "📤 Uploading: $SERVER_PATH"
echo "📁 To remote: $GDRIVE_PATH"
read -p "✅ Proceed with upload? (y/n): " CONFIRM

if [[ "$CONFIRM" != "y" ]]; then
    echo "❌ Upload cancelled."
    exit 0
fi

# === Upload logic ===
if [ -f "$SERVER_PATH" ]; then
    echo "📄 Detected file — using rclone copyto (with overwrite)"
    rclone copyto "$SERVER_PATH" "${GDRIVE_PATH}/$(basename "$SERVER_PATH")" --progress --ignore-times

elif [ -d "$SERVER_PATH" ]; then
    echo "📁 Detected folder — preparing upload..."

    if [[ "$TYPE" == "result" && "$CKPT_CHOICE" == "without" ]]; then
        echo "🚫 Excluding all .ckpt files..."
        rclone copy "$SERVER_PATH" "${GDRIVE_PATH}/$(basename "$SERVER_PATH")" \
            --progress --ignore-times \
            --exclude "*.ckpt"
    else
        echo "📁 Uploading all contents (including checkpoints)"
        rclone copy "$SERVER_PATH" "${GDRIVE_PATH}/$(basename "$SERVER_PATH")" \
            --progress --ignore-times
    fi

else
    echo "❌ Error: '$SERVER_PATH' does not exist"
    exit 1
fi