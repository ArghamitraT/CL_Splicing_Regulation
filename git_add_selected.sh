#!/bin/bash
# ~/git_quickpush.sh

exts=("py" "sh" "yaml" "yml" "ipynb")

echo "Adding files with extensions: ${exts[@]}"
for ext in "${exts[@]}"; do
    find . -type f -name "*.${ext}" -exec git add {} \;
done

echo "✅ Files added."

# Default commit message (or use the argument you pass)
msg=${1:-"update"}
git commit -m "$msg"
git push



# usage: bash git_quickpush.sh "minor update to lit.py"
