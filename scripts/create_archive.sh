#!/bin/bash
# Create a clean archive respecting .gitignore (without using git)

set -e

PROJECT_NAME="yamlllm"
OUTPUT_FILE="${PROJECT_NAME}.zip"

echo "Creating archive: $OUTPUT_FILE"
echo "Excluding patterns from .gitignore..."

# Remove old archive if exists
rm -f "$OUTPUT_FILE"

# Create zip excluding common patterns from .gitignore
# NOTE: outputs/ directory is INCLUDED (contains example outputs)
zip -r "$OUTPUT_FILE" . \
    -x "*.zip" \
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

echo ""
echo "✅ Archive created: $OUTPUT_FILE"
echo "Total size: $(du -h "$OUTPUT_FILE" | cut -f1)"
echo ""
echo "First 30 files in archive:"
unzip -l "$OUTPUT_FILE" | head -35
