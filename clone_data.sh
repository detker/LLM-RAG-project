#!/bin/bash

DOWNLOAD_DIR="${PWD}/data"

URLS=(
  "https://raw.githubusercontent.com/brandonstarxel/chunking_evaluation/refs/heads/main/chunking_evaluation/evaluation_framework/general_evaluation_data/corpora/chatlogs.md"
  "https://raw.githubusercontent.com/brandonstarxel/chunking_evaluation/refs/heads/main/chunking_evaluation/evaluation_framework/general_evaluation_data/corpora/state_of_the_union.md"
  "https://raw.githubusercontent.com/brandonstarxel/chunking_evaluation/refs/heads/main/chunking_evaluation/evaluation_framework/general_evaluation_data/corpora/wikitexts.md"
  "https://raw.githubusercontent.com/brandonstarxel/chunking_evaluation/refs/heads/main/chunking_evaluation/evaluation_framework/general_evaluation_data/questions_df.csv"
)

echo "Creating directory '$DOWNLOAD_DIR' (if it doesn't exist)..."
mkdir -p "$DOWNLOAD_DIR"

if [ $? -ne 0 ]; then
    echo "Error: Failed to create directory '$DOWNLOAD_DIR'"
    exit 1
fi

for url in "${URLS[@]}"; do
    filename=$(basename "$url")
    output_path="$DOWNLOAD_DIR/$filename"
    
    echo "Downloading: $filename"
    echo "From: $url"
    echo "To: $output_path"
    
    curl -L --fail --progress-bar -o "$output_path" "$url"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to download $url"
    else
        echo "Success: Downloaded $filename"
    fi
    echo "----------------------------------------"
done

echo "Download process completed."

echo "Summary:"
echo "  - Download directory: $DOWNLOAD_DIR"
echo "  - Files attempted: ${#URLS[@]}"
echo "  - Download time: $(date +"%Y-%m-%d %H:%M:%S")"

exit 0

