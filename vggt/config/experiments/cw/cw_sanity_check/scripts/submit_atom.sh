#!/bin/bash
#SBATCH --job-name=check_atom
#SBATCH --account=fair_amaia_cw_scale
#SBATCH --qos=scale
#SBATCH --partition=learn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=12:00:00

export PYTHONUNBUFFERED=1

python check_complete_aws.py --data_dir=s3://dino/datasets/repligen/dec/atom/