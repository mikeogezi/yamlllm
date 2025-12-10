#!/bin/bash
# Create a clean archive respecting .gitignore (without using git)

set -e

ARCHIVE_TYPE=${1:-"tar"}

PROJECT_NAME="yamlllm"
OUTPUT_FILE="${PROJECT_NAME}.${ARCHIVE_TYPE}"

echo "Creating archive: $OUTPUT_FILE"
echo "Excluding patterns from .gitignore..."

# Remove old archive if exists
rm -f "$OUTPUT_FILE"

# Create ${ARCHIVE_TYPE} excluding common patterns from .gitignore
# NOTE: outputs/ directory is INCLUDED (contains example outputs)
if [ "$ARCHIVE_TYPE" = "tar" ]; then
    # For tar, use -c (create), -f (file), and --exclude for exclusions
    tar -cf "$OUTPUT_FILE" \
        --exclude="*.${ARCHIVE_TYPE}" \
        --exclude="*/__pycache__/*" \
        --exclude="*.pyc" \
        --exclude="*.pyo" \
        --exclude="*/.pytest_cache/*" \
        --exclude="*/.mypy_cache/*" \
        --exclude="*.egg-info/*" \
        --exclude="*/.gemini/*" \
        --exclude=".gemini/*" \
        --exclude=".git/*" \
        --exclude=".gitignore" \
        --exclude="*.swp" \
        --exclude="*~" \
        --exclude=".DS_Store" \
        .
elif [ "$ARCHIVE_TYPE" = "zip" ]; then
    # For zip, use zip command with exclusions
    zip -r "$OUTPUT_FILE" . \
        -x "*.${ARCHIVE_TYPE}" \
        -x "*/__pycache__/*" \
        -x "*.pyc" \
        -x "*.pyo" \
        -x "*/.pytest_cache/*" \
        -x "*/.mypy_cache/*" \
        -x "*.egg-info/*" \
        -x "*/.gemini/*" \
        -x ".gemini/*" \
        -x ".git/*" \
        -x ".gitignore" \
        -x "*.swp" \
        -x "*~" \
        -x ".DS_Store"
else
    echo "Unsupported archive type: $ARCHIVE_TYPE"
    exit 1
fi

echo ""
echo "✅ Archive created: $OUTPUT_FILE"
echo "Total size: $(du -h "$OUTPUT_FILE" | cut -f1)"
echo ""
echo "First 30 files in archive:"
if [ "$ARCHIVE_TYPE" = "tar" ]; then
    tar -tf "$OUTPUT_FILE" | head -35
elif [ "$ARCHIVE_TYPE" = "zip" ]; then
    unzip -l "$OUTPUT_FILE" | head -35
else
    echo "Unknown archive type for listing"
fi
