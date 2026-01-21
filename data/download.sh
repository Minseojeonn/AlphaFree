#!/bin/bash

if ! command -v gdown &> /dev/null; then
    echo "gdown not found. Installing..."
    pip install --user gdown
fi

declare -A FILES
FILES["steam.zip"]="1BuEJPAZFnLTjrC8x0M5RvhRdTxq4qnT0"
FILES["amazon_movie.zip"]="10P4C2DF8XqJVkxQlkA2kqzzG1ZNdkrgZ"
FILES["amazon_book_2014.zip"]="1oHc-5aFqJnD9U2__r-gpfmV-k0dzMJYK"
FILES["amazon_video.zip"]="1pdhQLBhyrIrQV4BwYAn5S0RQslg8RKow"
FILES["amazon_baby.zip"]="1Oeud32uXAYaRRvCotXWkF5NZQNOrNiEi"
FILES["amazon_beauty_personal.zip"]="1pAwMK52yQD25efL1oKrSWrDInoH3QqvC"
FILES["amazon_health.zip"]="1YTL08brUJT7x_6SCHoGyGX_BCN6VGsxn"

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