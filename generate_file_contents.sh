#!/usr/bin/env bash
#
# generate_project_snapshot.sh
#
# Purpose:
#   1. Produce a top-level directory structure snapshot.
#   2. Capture the text contents of relevant files (e.g., .py, .sh, .yaml, .txt)
#      while excluding large/binary files, caches, virtual environment, and specified JSON files.
#
# Usage:
#   chmod +x generate_project_snapshot.sh
#   ./generate_project_snapshot.sh
#
# Result:
#   A file named "pipeline_snapshot.md" in the current directory, containing:
#     - A tree-based directory structure.
#     - The text contents of allowed file types.

DOC_FILE="pipeline_snapshot.md"

# Create or clear the output file
echo "# Project Snapshot" > "$DOC_FILE"
echo >> "$DOC_FILE"

#####################################
# 1) Directory Structure
#####################################

echo "## Directory Structure" >> "$DOC_FILE"
echo '```plaintext' >> "$DOC_FILE"
# Exclude patterns, including mlruns_backup_* to ignore backup directories
EXCLUDE_REGEX="_experiments_| __DEPRECATED|mlrun_*|mlruns_backup_*|venv|__pycache__|*.js|*.git|*.pytest_cache|mlruns|*.pth|*.pyc|*.png|*.jpg|*.jpeg|*.gif|*.bmp|*.tiff|*.xml|*.jgw|*.asc|*.deb|*.pdf|*.log"
tree -f -I "$EXCLUDE_REGEX" >> "$DOC_FILE" 2>/dev/null
echo '```' >> "$DOC_FILE"
echo >> "$DOC_FILE"

#####################################
# 2) Capture File Contents
#####################################
# Exclude known binary/large file types, logs, caches, virtual environment, and the two unwanted JSON files.

while IFS= read -r file; do
    # Skip directories
    if [ -d "$file" ]; then
        continue
    fi

    # Derive a short name for display
    file_name=$(basename "$file")

    # Decide syntax highlighting for code blocks
    case "$file" in
        *.py)   code_block="python" ;;
        *.sh)   code_block="bash" ;;
        *.yaml|*.yml) code_block="yaml" ;;
        *.toml) code_block="toml" ;;
        *.json) code_block="json" ;;
        *.txt)  code_block="plaintext" ;;
        *)      code_block="plaintext" ;;
    esac

    echo "## $file_name" >> "$DOC_FILE"
    echo "\`\`\`$code_block" >> "$DOC_FILE"

    # Append file contents
    cat "$file" >> "$DOC_FILE" 2>/dev/null

    echo "\`\`\`" >> "$DOC_FILE"
    echo >> "$DOC_FILE"

done < <(find . -type f \
    ! -path "*/__pycache__/*" \
    ! -path "*/.git/*" \
    ! -path "*/.pytest_cache/*" \
    ! -path "*/mlruns/*" \
    ! -path "*/mlruns_backup_*/*" \
    ! -path "*/venv/*" \
    ! -name "*.pth" \
    ! -name "*.zip" \
    ! -name "*.pyc" \
    ! -name "mlruns_*/*" \
    ! -name "*.pyo" \
    ! -name "*.pyd" \
    ! -name "*.so" \
    ! -name "*.DS_Store" \
    ! -name "*.pdf" \
    ! -name "*.png" \
    ! -name "*.jpg" \
    ! -name "*.jpeg" \
    ! -name "*.gif" \
    ! -name "*.bmp" \
    ! -name "*.tiff" \
    ! -name "*.xml" \
    ! -name "*.jgw" \
    ! -name "*.asc" \
    ! -name "*.deb" \
    ! -name "*.log" \
    ! -name "*.md" \
    ! -name "coordinate_mapping.json" \
    ! -name "split_mapping.json" \
    ! -name "_experiments_/*" \
    ! -name "*.json" \
    ! -name "*.csv" \
    ! -name "__DEPRECATED/*" \
    2>/dev/null)

echo "Documentation generated in $DOC_FILE."
