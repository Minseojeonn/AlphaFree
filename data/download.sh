#!/bin/bash

if ! command -v gdown &> /dev/null; then
    echo "gdown not found. Installing..."
    pip install --user gdown
fi

declare -A FILES
FILES["steam.zip"]="1Ck5wx2wTxFFaUoZNSG8dA-pyHEMD1ryA"
FILES["amazon_movie.zip"]="169UHsoOElhwL1LkYnBCpi6kdJ0hmFBVV"
FILES["amazon_book_2014.zip"]="1PkD5sT7yy8qDNqUSKT8wkZZYUL9U4u5A"
FILES["amazon_video.zip"]="1rYbqJPjvHhac92Gx2A25tHtwWKCuYsIe"
FILES["amazon_baby.zip"]="18NmIghILo7EKYD94DE0SEl7M8YnanElD"
FILES["amazon_beauty_personal.zip"]="1M_ybwPrYxseYZwhzxevvCP0zPQO2gJEA"
FILES["amazon_health.zip"]="1LFton2fZx9yDcTBsIB9pmiHyGKFuU4re"
echo "Select a file to download:"
select NAME in "${!FILES[@]}"; do
    if [[ -n "$NAME" ]]; then
        BASE_NAME="${NAME%.zip}"
        FILE_ID="${FILES[$NAME]}"
        DEST="./$NAME"
        EXTRACT_DIR="./$BASE_NAME"

        # download
        echo "Downloading $NAME using gdown..."
        gdown --id "$FILE_ID" -O "$DEST"

        #unzip 
        echo "Extracting $NAME to temp folder..."
        TEMP_DIR="./temp/$BASE_NAME"
        mkdir -p "$TEMP_DIR"
        unzip -o "$DEST" -d "$TEMP_DIR"
        rm "$DEST"

        # move file
        mv "$TEMP_DIR/$BASE_NAME" "./"

        # cleanup
        rm -rf ./temp 

        echo "Done. $NAME is available in $EXTRACT_DIR"
        break
    else
        echo "Invalid selection. Try again."
    fi
done
