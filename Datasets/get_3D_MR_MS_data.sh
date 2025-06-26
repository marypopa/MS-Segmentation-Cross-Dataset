#!/bin/bash

# Exit script on any error
set -e

# Create base directory
mkdir -p 3D-MR-MS
cd 3D-MR-MS

# List of zip files to download
urls=(
    "https://lit.fe.uni-lj.si/data/research/resources/3D-MR-MS/3D-MR-MS_patient01-05.zip"
    "https://lit.fe.uni-lj.si/data/research/resources/3D-MR-MS/3D-MR-MS_patient06-10.zip"
    "https://lit.fe.uni-lj.si/data/research/resources/3D-MR-MS/3D-MR-MS_patient11-15.zip"
    "https://lit.fe.uni-lj.si/data/research/resources/3D-MR-MS/3D-MR-MS_patient16-20.zip"
    "https://lit.fe.uni-lj.si/data/research/resources/3D-MR-MS/3D-MR-MS_patient21-25.zip"
    "https://lit.fe.uni-lj.si/data/research/resources/3D-MR-MS/3D-MR-MS_patient26-30.zip"
)

# Download each zip file
for url in "${urls[@]}"; do
    wget "$url"
done

# Create a directory for all patients
mkdir -p patients

# Unzip each file into a temp folder and move contents into 'patients'
for zip in *.zip; do
    temp_dir="${zip%.zip}"
    mkdir "$temp_dir"
    unzip "$zip" -d "$temp_dir"
    mv "$temp_dir"/* patients/
    rm -r "$temp_dir"
done

echo "All 3D-MR-MS dataset files downloaded, unzipped, and organized in 3D-MR-MS/patients"
