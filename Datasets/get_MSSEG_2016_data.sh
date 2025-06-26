#!/bin/bash

# Access to MSSEG-2016 data can be obtained using the following link
# https://shanoir.irisa.fr/shanoir-ng/account/study/209/account-request?study=MSSEG%202016&function=consumer
# Download the MSSEG-Training and MSSEG-Testing archives in the MSSEG-2016 directory

set -euo pipefail

# Script to prepare MSSEG-2016 dataset directory structure

echo "Starting setup for MSSEG-2016..."

# Dataset link information
echo "Access to MSSEG-2016 data requires registration:"
echo "https://shanoir.irisa.fr/shanoir-ng/account/study/209/account-request?study=MSSEG%202016&function=consumer"
echo "Download the MSSEG-Training and MSSEG-Testing archives in the MSSEG-2016 directory"

# Create the target directory if it doesn't exist
TARGET_DIR="MSSEG-2016"
if [ ! -d "$TARGET_DIR" ]; then
    echo "Creating directory: $TARGET_DIR"
    mkdir "$TARGET_DIR"
else
    echo "Directory $TARGET_DIR already exists."
fi

cd "$TARGET_DIR"

# Define expected archive files
FILES=("MSSEG-Training.zip" "MSSEG-Testing.zip")

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

echo "MSSEG-2016 setup complete."
