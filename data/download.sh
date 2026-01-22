#!/bin/bash

if ! command -v gdown &> /dev/null; then
    echo "gdown not found. Installing..."
    pip install --user gdown
fi

declare -A FILES
FILES["steam.zip"]="1pUaW4cl56VpwtH6ICCKn_HQUlqqN71KL"
FILES["amazon_movie.zip"]="10P4C2DF8XqJVkxQlkA2kqzzG1ZNdkrgZ"
FILES["amazon_book_2014.zip"]="1oHc-5aFqJnD9U2__r-gpfmV-k0dzMJYK"
FILES["amazon_video.zip"]="1BUm2k4Yn2oPLgQTzLvQncgPZqpvmCT7j"
FILES["amazon_baby.zip"]="1D3ukli7JKVY5zv5Km3Q8bvkaTEt0KBUJ"
FILES["amazon_beauty_personal.zip"]="1TY4iWVJQZ1DgVc74ZwQz5k1aRrxfDW0Z"
FILES["amazon_health.zip"]="1ZBryoZ6rO6lQ9fID8Fw1_xD-qTW4187y"
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