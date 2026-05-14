#!/bin/bash

# Source directory
SRC_DIR="/checkpoint/repligen/jianyuan/data/shier_v1"

# Destination directories
DEST1="/home/jianyuan/src/omega/projects/vggt/data/valid_seqs"
DEST2="/mnt/coreai_3d/tree/jianyuan/valid_seqs"

# Create destination directories if they don't exist
mkdir -p "$DEST1"
mkdir -p "$DEST2"

# Loop through all folders in the source directory
for folder in "$SRC_DIR"/*/; do
    # Get the folder name (basename)
    folder_name=$(basename "$folder")
    
    # Create the txt filename
    txt_file="${folder_name}.txt"
    
    echo "Processing: $folder_name"
    
    # List contents and save to txt file
    ls "$folder" > "$txt_file"
    
    # Copy to both destinations
    cp "$txt_file" "$DEST1/$txt_file"
    cp "$txt_file" "$DEST2/$txt_file"
    
    # Remove the temporary local file
    rm "$txt_file"
    
    echo "  -> Copied to $DEST1/$txt_file"
    echo "  -> Copied to $DEST2/$txt_file"
done

echo "Done!"