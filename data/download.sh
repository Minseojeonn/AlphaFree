#!/bin/bash

# Check if gdown is installed, if not, install it via pip
if ! command -v gdown &> /dev/null; then
    echo "gdown not found. Installing..."
    pip install --user gdown
fi

# Define dataset names and their corresponding Google Drive file IDs
declare -A FILES
FILES["steam"]="1Ck5wx2wTxFFaUoZNSG8dA-pyHEMD1ryA"
FILES["amazon_movie"]="169UHsoOElhwL1LkYnBCpi6kdJ0hmFBVV"
FILES["amazon_book_2014"]="1PkD5sT7yy8qDNqUSKT8wkZZYUL9U4u5A"
FILES["amazon_video"]="1rYbqJPjvHhac92Gx2A25tHtwWKCuYsIe"
FILES["amazon_baby"]="18NmIghILo7EKYD94DE0SEl7M8YnanElD"
FILES["amazon_beauty_personal"]="1M_ybwPrYxseYZwhzxevvCP0zPQO2gJEA"
FILES["amazon_health"]="1LFton2fZx9yDcTBsIB9pmiHyGKFuU4re"

SELECTED_DATASET=""

# 1. Parse command-line arguments (e.g., --dataset steam)
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) SELECTED_DATASET="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Function to handle download, extraction, and cleanup
download_and_extract() {
    local BASE_NAME=$1
    local FILE_ID=${FILES[$BASE_NAME]}
    
    # Check if the requested dataset exists in the dictionary
    if [[ -z "$FILE_ID" ]]; then
        echo "Error: Dataset '$BASE_NAME' not found in the list."
        echo "Available datasets: ${!FILES[@]}"
        exit 1
    fi

    local DEST="./${BASE_NAME}.zip"
    local TEMP_DIR="./temp/$BASE_NAME"

    # Start downloading using gdown
    echo "Downloading $BASE_NAME using gdown..."
    gdown --id "$FILE_ID" -O "$DEST"

    # Extract the zip file to a temporary directory
    echo "Extracting $DEST to temp folder..."
    mkdir -p "$TEMP_DIR"
    unzip -o "$DEST" -d "$TEMP_DIR"
    
    # Remove the zip file after extraction
    rm "$DEST"

    # Move the extracted data to the current directory and cleanup
    mv "$TEMP_DIR/$BASE_NAME" "./"
    rm -rf ./temp 
    
    echo "Done. $BASE_NAME is now available in ./${BASE_NAME}"
}

# 2. Execution Logic
if [[ -n "$SELECTED_DATASET" ]]; then
    # Case A: Execute immediately if an argument is provided
    download_and_extract "$SELECTED_DATASET"
else
    # Case B: Provide an interactive menu if no argument is given
    echo "No dataset specified. Please select a file to download:"
    select NAME in "${!FILES[@]}"; do
        if [[ -n "$NAME" ]]; then
            download_and_extract "$NAME"
            break
        else
            echo "Invalid selection. Please try again."
        fi
    done
fi