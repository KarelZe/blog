#!/bin/bash
# Process markdown files with Pandoc for citation resolution
# Usage: ./scripts/process-citations.sh

set -e  # Exit on error

CONTENT_DIR="content"
BIBLIOGRAPHY="library.bib"
CSL_STYLE="${CSL_STYLE:-chicago-author-date.csl}"

echo "Processing citations in content files..."
echo "Bibliography: $BIBLIOGRAPHY"
echo "CSL Style: $CSL_STYLE"
echo ""

# Check if Pandoc is installed
if ! command -v pandoc &> /dev/null; then
    echo "Error: Pandoc is not installed. Please install it first."
    exit 1
fi

# Check if bibliography file exists
if [ ! -f "$BIBLIOGRAPHY" ]; then
    echo "Error: Bibliography file '$BIBLIOGRAPHY' not found."
    exit 1
fi

# Check if CSL style file exists
if [ ! -f "$CSL_STYLE" ]; then
    echo "Error: CSL style file '$CSL_STYLE' not found."
    exit 1
fi

# Counter for processed files
processed=0

# Find all .md.draft files that contain Pandoc citations [@...]
find "$CONTENT_DIR" -name "*.md.draft" -type f | while read -r draft_file; do
    if grep -q '\[@[^]]*\]' "$draft_file"; then
        # Derive the output .md file name
        output_file="${draft_file%.draft}"

        echo "Processing: $draft_file → $output_file"

        # Create temporary output
        temp_file="${output_file}.tmp"

        # Process with Pandoc: resolve citations and generate bibliography
        # Using -fenced_divs to disable the ::: syntax
        # Using --standalone to preserve YAML frontmatter
        # Using +raw_attribute to preserve Hugo shortcodes
        pandoc "$draft_file" \
            --citeproc \
            --bibliography="$BIBLIOGRAPHY" \
            --csl="$CSL_STYLE" \
            --from=markdown \
            --to=markdown-fenced_divs+raw_attribute \
            --wrap=preserve \
            --standalone \
            --output="$temp_file"

        # Post-process to unescape Hugo shortcodes
        # Pandoc escapes {{ < to {{\< and > }} to \>}}
        sed -i '' 's/{{\\</{{</g; s/\\>}}/>}}/g' "$temp_file"

        # Replace output file if successful
        if [ $? -eq 0 ]; then
            mv "$temp_file" "$output_file"
            processed=$((processed + 1))
        else
            echo "Error processing $draft_file"
            rm -f "$temp_file"
            exit 1
        fi
    fi
done

echo ""
echo "Citation processing complete! Processed $processed file(s)."
echo "Source files (.md.draft) preserved with unresolved citations."
