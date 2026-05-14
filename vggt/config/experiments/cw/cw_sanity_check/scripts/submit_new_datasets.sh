#!/bin/bash

# List of datasets to check
datasets=(
    "tum_dynamic"
    "nrgbd"
    "dycheck"
    "da3_dtu"
    "da3_eth3d"
    "da3_hiroom"
    "da3_scannetpp"
    "da3_sevenscenes"
    "tartanground"
)

# Base directory for dataset configs
CONFIG_DIR="/home/jianyuan/src/omega/projects/vggt/config/experiments/cw/datasets"

# Script to submit single job
SUBMIT_SCRIPT="/home/jianyuan/src/omega/projects/vggt/config/experiments/cw/cw_sanity_check/scripts/submit_yaml.py"

for dataset in "${datasets[@]}"; do
    yaml_file="${CONFIG_DIR}/${dataset}.yaml"
    if [ -f "$yaml_file" ]; then
        echo "Submitting check for ${dataset}..."
        python "$SUBMIT_SCRIPT" "$yaml_file"
    else
        echo "Warning: Config file for ${dataset} not found at ${yaml_file}"
    fi
done




# 
