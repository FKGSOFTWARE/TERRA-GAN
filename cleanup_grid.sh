#!/usr/bin/env bash
#
# cleanup_grid.sh
# Asks for a grid square ID (e.g. NJ05). Moves all matching
# files/directories to _to_delete_, then empties that folder on a second run.

TO_DELETE_DIR="_to_delete_"

# 1) If _to_delete_ exists and has leftover items, remove them first
if [ -d "$TO_DELETE_DIR" ] && [ "$(ls -A "$TO_DELETE_DIR")" ]; then
    echo "Removing old contents of $TO_DELETE_DIR..."
    rm -rf "$TO_DELETE_DIR"/*
fi

# 2) Prompt for grid square ID
read -p "Enter Grid Square ID (e.g. NJ05): " GRID_ID
mkdir -p "$TO_DELETE_DIR"

# Build a case-insensitive regex that matches:
#   - "NJ05" at the end of a name, or
#   - "NJ05" followed by any NON-digit character.
#
# Example matches: "NJ05", "NJ05_", "NJ05abc", but NOT "NJ050" or "NJ051"
REGEX=".*${GRID_ID}([^0-9].*|$)"

echo "Searching for items matching '$GRID_ID' not followed by a digit ..."
echo "Moving all matches into $TO_DELETE_DIR ..."

# 3) Find matching files/directories; exclude the logs folder and
#    _to_delete_ folder so we don't move or re-move those.
#    Use -depth so we move child items first, then the parent directory.
find . -depth \
  -path "./logs" -prune -o \
  -path "./${TO_DELETE_DIR}" -prune -o \
  -regextype posix-extended -iregex "$REGEX" \
  -exec mv {} "${TO_DELETE_DIR}"/ \; 2>/dev/null

# 4) Done. Show the user what got moved into _to_delete_
echo
echo "Done. The following items were moved into ${TO_DELETE_DIR}:"
ls -R "${TO_DELETE_DIR}"
