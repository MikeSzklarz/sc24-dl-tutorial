#!/bin/bash

# Set the base URL and target directory
BASE_URL="https://portal.nersc.gov/project/dasrepo/pharring/sc23_data/"
TARGET_DIR="data"
LOG_FILE="download_log.txt"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Redirect stdout and stderr to a log file
exec > >(awk 'NR % 10 == 0' > "$LOG_FILE") 2>&1

echo "Starting download at $(date)"
echo "Base URL: $BASE_URL"
echo "Target directory: $TARGET_DIR"
echo ""

# Use wget to download the files, preserving the folder structure, without overwriting existing files
wget -r -np -nH --cut-dirs=4 -P "$TARGET_DIR" -R "index.html*" -nc "$BASE_URL"

echo ""
echo "Download completed at $(date). Data saved in '$TARGET_DIR'."
