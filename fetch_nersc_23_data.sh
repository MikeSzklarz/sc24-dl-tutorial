#!/bin/bash

# Set the base URL and target directory
BASE_URL="https://portal.nersc.gov/project/dasrepo/pharring/sc23_data/"
TARGET_DIR="data"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Use wget to download the files, preserving the folder structure, without overwriting existing files
wget -r -np -nH --cut-dirs=4 -P "$TARGET_DIR" -R "index.html*" -nc "$BASE_URL"

echo "Download completed. Data saved in '$TARGET_DIR'."
