#!/bin/bash

# Reset the logs directory for a clean start
ehco "Resetting logs directory"
echo "Removed logs"
rm -r logs 

# Create the logs directory if it doesn't exist
# and create a subdirectory for slurm logs
echo "Creating logs directory"
mkdir -p logs
mkdir -p logs/slurm