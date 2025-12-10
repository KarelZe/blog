#!/bin/bash
# Clean up unreferenced files in a Hugo page bundle
# Usage: ./scripts/clean-pagebundle.sh <path-to-page-bundle> [--delete]

set -e

BUNDLE_DIR="$1"
DELETE_MODE="${2:-}"

if [ -z "$BUNDLE_DIR" ]; then
    echo "Usage: $0 <path-to-page-bundle> [--delete]"
    echo "Example: $0 content/post/goldfish-loss/"
    echo ""
    echo "Options:"
    echo "  --delete    Actually delete files (default is dry-run)"
    exit 1
fi

if [ ! -d "$BUNDLE_DIR" ]; then
    echo "Error: Directory '$BUNDLE_DIR' does not exist"
    exit 1
fi

# Check if index.md or index.md.draft exists
INDEX_FILE=""
if [ -f "$BUNDLE_DIR/index.md" ]; then
    INDEX_FILE="$BUNDLE_DIR/index.md"
fi

DRAFT_FILE=""
if [ -f "$BUNDLE_DIR/index.md.draft" ]; then
    DRAFT_FILE="$BUNDLE_DIR/index.md.draft"
fi

if [ -z "$INDEX_FILE" ] && [ -z "$DRAFT_FILE" ]; then
    echo "Error: No index.md or index.md.draft found in $BUNDLE_DIR"
    exit 1
fi

echo "Analyzing page bundle: $BUNDLE_DIR"
echo "Reference files: $INDEX_FILE $DRAFT_FILE"
echo ""

if [ "$DELETE_MODE" != "--delete" ]; then
    echo "DRY RUN MODE (use --delete to actually remove files)"
    echo ""
fi

# Counter
unreferenced_count=0
deleted_count=0

# Find all files in the bundle directory
find "$BUNDLE_DIR" -type f | while read -r file; do
    # Get the basename
    basename=$(basename "$file")

    # Skip index files themselves
    if [ "$basename" = "index.md" ] || [ "$basename" = "index.md.draft" ]; then
        continue
    fi

    # Check if the file is referenced in either index.md or index.md.draft
    referenced=false

    if [ -n "$INDEX_FILE" ] && grep -q "$basename" "$INDEX_FILE"; then
        referenced=true
    fi

    if [ -n "$DRAFT_FILE" ] && grep -q "$basename" "$DRAFT_FILE"; then
        referenced=true
    fi

    # If not referenced, mark for deletion
    if [ "$referenced" = false ]; then
        echo "❌ Unreferenced: $file"
        unreferenced_count=$((unreferenced_count + 1))

        if [ "$DELETE_MODE" = "--delete" ]; then
            rm "$file"
            echo "   🗑️ DELETED"
            deleted_count=$((deleted_count + 1))
        fi
    fi
done

echo ""
if [ "$DELETE_MODE" = "--delete" ]; then
    echo "Deleted $deleted_count unreferenced file(s)"
else
    echo "Found $unreferenced_count unreferenced file(s)"
    echo "Run with --delete to remove them"
fi
