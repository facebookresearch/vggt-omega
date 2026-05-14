#!/bin/bash

# Fill this list with dataset yaml stems to submit.
datasets=(
    # "omniworld"
    # "scannet"
)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CW_DIR="$SCRIPT_DIR/../.."

DATASET_DIRS=(
    "$CW_DIR/datasets_real"
    "$CW_DIR/datasets_synt"
)

for dataset in "${datasets[@]}"; do
    found_any=0
    for dataset_dir in "${DATASET_DIRS[@]}"; do
        yaml_file="${dataset_dir}/${dataset}.yaml"
        if [ -f "$yaml_file" ]; then
            found_any=1
            echo "Submitting intrinsic check for ${yaml_file}..."
            python "$SCRIPT_DIR/submit_yaml.py" "$yaml_file"
        fi
    done

    if [ "$found_any" -eq 0 ]; then
        echo "Warning: Config file for ${dataset} not found in datasets_real or datasets_synt"
    fi
done
