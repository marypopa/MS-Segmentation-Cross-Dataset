

#!/bin/bash

# Access to ISBI-2015 data can be obtained using the following link
# https://smart-stats-tools.org/lesion-challenge
# Download the ISBI-2015 in the ISBI_2015 directory

set -euo pipefail

# Script to prepare ISBI-2015 dataset directory structure

echo "Starting setup for ISBI_2015..."

# Dataset link information
echo "Access to ISBI-2015 data requires registration:"
echo "https://smart-stats-tools.org/lesion-challenge"
echo

# Create the target directory if it doesn't exist
TARGET_DIR="ISBI_2015"
if [ ! -d "$TARGET_DIR" ]; then
    echo "Creating directory: $TARGET_DIR"
    mkdir "$TARGET_DIR"
else
    echo "Directory $TARGET_DIR already exists."
fi

cd "$TARGET_DIR"

# Define expected archive files
FILES=("training_final_v4.zip")

# Loop through and unzip if the archive exists
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        temp_dir="${file%.zip}"
        echo "Unzipping $file..."
		UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip "$file" -d "$temp_dir"
    else
        echo "Warning: File $file not found in the parent directory. Skipping."
    fi
done

echo "ISBI_2015 setup complete."
