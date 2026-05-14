#!/bin/bash
#SBATCH --account=fair_amaia_cw_scale
#SBATCH --qos=scale
#SBATCH --partition=learn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00

export PYTHONUNBUFFERED=1

# This script is intended to be called by sbatch from the scripts directory.
# It locates the python script in the parent directory.

if [ -z "$CHECK_SCRIPT" ]; then
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    PARENT_DIR="$(dirname "$SCRIPT_DIR")"
    CHECK_SCRIPT="$PARENT_DIR/check_complete_aws.py"
fi

if [ ! -f "$CHECK_SCRIPT" ]; then
    echo "Error: Cannot find check_complete_aws.py at $CHECK_SCRIPT"
    exit 1
fi

echo "Running sanity check..."
echo "Script: $CHECK_SCRIPT"
echo "Args: $@"

python "$CHECK_SCRIPT" "$@"

