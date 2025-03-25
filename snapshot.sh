#!/usr/bin/env bash
# snapshot.sh - Generate a Markdown snapshot of your project’s pruned tree and changed file contents.
# Requirements: git and tree must be installed. Run inside a git repository.

# Exit if not in a git repository.
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo "Error: Not inside a git repository."
    exit 1
fi

# Check for tree command.
if ! command -v tree &>/dev/null; then
    echo "Error: 'tree' command not found. Please install it."
    exit 1
fi

DOC_FILE="pipeline_snapshot.md"

# Clear/create the output file with a header.
{
  echo "# Project Snapshot"
  echo "Generated on $(date)"
  echo
  echo "## Directory Structure"
  echo '```plaintext'
} > "$DOC_FILE"

# Blacklist for tree output: prune unwanted directories and file types.
EXCLUDE_REGEX="(__pycache__|*\.git|*\.pytest_cache|mlruns|*\.pth|*\.pyc|*\.png|*\.jpg|*\.jpeg|*\.gif|*\.bmp|*\.tiff|*\.xml|*\.jgw|*\.asc|*\.deb|*\.pdf|*\.log)"

# Append the pruned directory tree.
tree -f -I "$EXCLUDE_REGEX" >> "$DOC_FILE" 2>/dev/null
{
  echo '```'
  echo
  echo "## Modified Files (since last commit)"
  echo
} >> "$DOC_FILE"

# Get list of files changed since last commit.
changed_files=$(git diff --name-only HEAD)

# Function: returns 0 if file is blacklisted.
is_blacklisted() {
  local file="$1"
  [[ "$file" =~ __pycache__ ]] || [[ "$file" =~ \.git/ ]] || [[ "$file" =~ __pytest_cache__ ]] || \
  [[ "$file" =~ mlruns ]] || [[ "$file" =~ \.pth$ ]] || [[ "$file" =~ \.zip$ ]] || \
  [[ "$file" =~ \.pyc$ ]] || [[ "$file" =~ \.pyo$ ]] || [[ "$file" =~ \.pyd$ ]] || \
  [[ "$file" =~ \.so$ ]] || [[ "$file" =~ \.DS_Store$ ]] || [[ "$file" =~ \.pdf$ ]] || \
  [[ "$file" =~ \.png$ ]] || [[ "$file" =~ \.jpe?g$ ]] || [[ "$file" =~ \.gif$ ]] || \
  [[ "$file" =~ \.bmp$ ]] || [[ "$file" =~ \.tiff$ ]] || [[ "$file" =~ \.xml$ ]] || \
  [[ "$file" =~ \.jgw$ ]] || [[ "$file" =~ \.asc$ ]] || [[ "$file" =~ \.deb$ ]] || \
  [[ "$file" =~ \.log$ ]] || [[ "$file" =~ \.md$ ]]
}

# Loop over each changed file.
for file in $changed_files; do
  # Only process if it exists and is a regular file.
  if [ ! -f "$file" ]; then
    continue
  fi

  # Skip if the file matches our blacklist.
  if is_blacklisted "$file"; then
    continue
  fi

  # Derive a simple name and decide on syntax highlighting.
  filename=$(basename "$file")
  case "$file" in
    *.py)   lang="python" ;;
    *.sh)   lang="bash" ;;
    *.yaml|*.yml) lang="yaml" ;;
    *.toml) lang="toml" ;;
    *.json) lang="json" ;;
    *.txt)  lang="plaintext" ;;
    *)      lang="plaintext" ;;
  esac

  {
    echo "### $file"
    echo "\`\`\`$lang"
    cat "$file"
    echo "\`\`\`"
    echo
  } >> "$DOC_FILE"
done

echo "Snapshot generated in $DOC_FILE"
