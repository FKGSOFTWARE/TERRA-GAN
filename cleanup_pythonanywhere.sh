#!/bin/bash
#
# cleanup_pythonanywhere.sh - Script to clean up PythonAnywhere files
#
# This script provides a convenient way to clean up annotation and image files
# on your PythonAnywhere server.

# Default options
DRY_RUN=false
ANNOTATIONS=false
IMAGES=false
GRID=""
FORCE=false

# Display help message
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -a, --annotations       Delete files in annotations directory"
    echo "  -i, --images            Delete files in images directory"
    echo "  -g, --grid GRID         Filter by grid square prefix (e.g., NH70)"
    echo "  -d, --dry-run           List files that would be deleted without actually deleting"
    echo "  -f, --force             Skip confirmation prompt"
    echo "  -h, --help              Show this help message"
    echo
    echo "Examples:"
    echo "  $0 --annotations --grid NH70       Delete all NH70 annotations"
    echo "  $0 --images --dry-run              Show all images that would be deleted"
    echo "  $0 --annotations --images --force  Delete all annotations and images without confirmation"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -a|--annotations)
            ANNOTATIONS=true
            shift
            ;;
        -i|--images)
            IMAGES=true
            shift
            ;;
        -g|--grid)
            GRID="$2"
            shift
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Build the command
CMD="python utils/api/pythonanywhere_cleanup.py"

if [ "$ANNOTATIONS" = true ]; then
    CMD+=" --annotations"
fi

if [ "$IMAGES" = true ]; then
    CMD+=" --images"
fi

if [ -n "$GRID" ]; then
    CMD+=" --grid $GRID"
fi

if [ "$DRY_RUN" = true ]; then
    CMD+=" --dry-run"
fi

if [ "$FORCE" = true ]; then
    CMD+=" --force"
fi

# If no actions were specified, show help
if [ "$ANNOTATIONS" = false ] && [ "$IMAGES" = false ]; then
    echo "Error: You must specify at least one action (--annotations or --images)"
    show_help
    exit 1
fi

# Show the command that will be executed
echo "Executing: $CMD"

# Execute the command
eval $CMD

# Report completion
if [ $? -eq 0 ]; then
    echo "Cleanup completed successfully"
else
    echo "Cleanup encountered errors, check the logs for details"
    exit 1
fi
