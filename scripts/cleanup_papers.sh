#!/bin/bash
# Clean up papers folder - keep only tetc_paper.tex and required class files

cd /workspaces/quantoniumos/papers

# Files to KEEP
KEEP_FILES=(
    "tetc_paper.tex"
    "IEEEtran.cls"
    "IEEEtran.bst"
    "quantoniumos_rft.bib"
)

echo "=== Cleaning papers/ folder ==="
echo "Keeping: ${KEEP_FILES[*]}"
echo ""

# Create backup directory
mkdir -p ../papers_backup

# Move files to backup
for f in *.tex *.pdf *.md *.json; do
    if [[ -f "$f" ]]; then
        keep=false
        for k in "${KEEP_FILES[@]}"; do
            if [[ "$f" == "$k" ]]; then
                keep=true
                break
            fi
        done
        
        if [[ "$keep" == false ]]; then
            echo "  Moving to backup: $f"
            mv "$f" ../papers_backup/
        else
            echo "  Keeping: $f"
        fi
    fi
done

echo ""
echo "=== Remaining files in papers/ ==="
ls -la

echo ""
echo "Backup location: papers_backup/"
echo "Done!"
