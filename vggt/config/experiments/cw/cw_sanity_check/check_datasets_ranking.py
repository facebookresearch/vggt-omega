import os
import glob
import yaml
import random
import fsspec
import logging
import argparse
import concurrent.futures
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_fs(path: str):
    if path.startswith("s3://"):
        return fsspec.filesystem("s3", profile="dino"), path, True
    return fsspec.filesystem("file"), path, False

def check_dataset(yaml_path, datasets_dir):
    dataset_name = os.path.basename(yaml_path).replace('.yaml', '')
    
    with open(yaml_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            logging.error(f"Error parsing {yaml_path}: {exc}")
            return

    if config is None:
        print(f"Skipping empty config: {yaml_path}")
        return

    unified_dir = config.get("UNIFIED_DIR")
    if not unified_dir:
        # logging.warning(f"Skipping {dataset_name}: UNIFIED_DIR not found in config")
        return

    sample_by_index = config.get("sample_by_index", False)
    
    fs, path, is_s3 = get_fs(unified_dir)
    
    try:
        # List scenes
        path = path.rstrip("/")
        # Get list of files/directories in the unified directory
        items = fs.ls(path, detail=True)
        
        # Filter for directories (scenes)
        scenes = [item['name'] for item in items if item['type'] == 'directory']
        
        if not scenes:
            logging.warning(f"Dataset {dataset_name} is empty (no scenes found in {unified_dir})")
            return

        # Pick random scene
        random_scene = random.choice(scenes)
        
        # Check for ranking.npy
        ranking_path = os.path.join(random_scene, "ranking.npy")
        
        exists = fs.exists(ranking_path)
        
        if not exists:
            if not sample_by_index:
                 print(f"DATASET ISSUE: {dataset_name}")
                 logging.info(f"  -> Detail: Scene '{os.path.basename(random_scene)}' missing ranking.npy AND sample_by_index is {sample_by_index}")
        
    except Exception as e:
        logging.error(f"Error checking {dataset_name} at {unified_dir}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Check dataset configs for ranking.npy and sample_by_index")
    parser.add_argument("--datasets_dir", type=str, default="projects/vggt/config/experiments/cw/datasets", help="Path to datasets config directory")
    parser.add_argument("--workers", type=int, default=16, help="Number of worker threads")
    args = parser.parse_args()

    # Expand user path if necessary
    datasets_dir = os.path.expanduser(args.datasets_dir)
    if not os.path.exists(datasets_dir):
        # Try relative to current working directory or known project root
        project_root = "/home/jianyuan/src/omega"
        datasets_dir = os.path.join(project_root, args.datasets_dir)

    if not os.path.exists(datasets_dir):
         print(f"Error: Datasets directory not found: {datasets_dir}")
         return

    yaml_files = glob.glob(os.path.join(datasets_dir, "*.yaml"))
    logging.info(f"Found {len(yaml_files)} dataset configs in {datasets_dir}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(check_dataset, yaml_file, datasets_dir) for yaml_file in yaml_files]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(yaml_files)):
            pass

if __name__ == "__main__":
    main()

