#!/bin/bash
# Add only selected file extensions recursively to git

# Define extensions you want to include
exts=("py" "sh" "yaml" "yml" "ipynb" "txt")

echo "Adding files with extensions: ${exts[@]}"
for ext in "${exts[@]}"; do
    find . -type f -name "*.${ext}" -exec git add {} \;
done

echo "✅ Files added to staging area."

