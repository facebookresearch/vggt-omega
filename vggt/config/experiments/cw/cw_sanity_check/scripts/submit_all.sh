#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Directory containing dataset yamls, relative to the script location
DATASETS_DIR="$SCRIPT_DIR/../../datasets"

if [ ! -d "$DATASETS_DIR" ]; then
    echo "Error: Datasets directory not found at $DATASETS_DIR"
    exit 1
fi

echo "Found datasets dir: $DATASETS_DIR"

for yaml_file in "$DATASETS_DIR"/*.yaml; do
    if [ -f "$yaml_file" ]; then
        # Run submit_yaml.py using absolute path
        python "$SCRIPT_DIR/submit_yaml.py" "$yaml_file"
    fi
done
