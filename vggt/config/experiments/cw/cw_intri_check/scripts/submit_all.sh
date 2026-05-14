#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CW_DIR="$SCRIPT_DIR/../.."

DATASET_DIRS=(
    "$CW_DIR/datasets_real"
    "$CW_DIR/datasets_synt"
)

for dataset_dir in "${DATASET_DIRS[@]}"; do
    if [ ! -d "$dataset_dir" ]; then
        echo "Warning: Dataset directory not found at $dataset_dir"
        continue
    fi

    echo "Scanning dataset dir: $dataset_dir"
    for yaml_file in "$dataset_dir"/*.yaml; do
        if [ -f "$yaml_file" ]; then
            python "$SCRIPT_DIR/submit_yaml.py" "$yaml_file"
        fi
    done
done
