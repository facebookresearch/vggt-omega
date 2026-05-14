import argparse
import logging
import os
import sys
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
import fsspec
from io import BytesIO
import concurrent.futures

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

COMPLETION_INDICATOR_FILE = "complete_log.txt"
IMAGE_FOLDER_NAME = "images"
DEPTH_FOLDER_NAME = "depths"
DEPTH_MASKS_FOLDER_NAME = "depth_masks"
MIN_FILE_SIZE_BYTES = 1024  # Minimum file size to consider valid (0.1 MB)


def get_fs(path: str):
    if path.startswith("s3://"):
        return fsspec.filesystem("s3", profile="dino"), path, True
    return fsspec.filesystem("file"), path, False


def find_scenes(data_dir: str, fs, scene_list_file: Path | None = None) -> list[str]:
    """
    Find scene directories.
    If scene_list_file is provided, it reads scene names from the file.
    Otherwise, it finds all scene directories in data_dir that have a completion indicator.
    """
    scenes = []
    logging.info(f"Searching for scenes in {data_dir}...")
    
    if not fs.exists(data_dir):
        logging.warning(f"Data directory {data_dir} does not exist.")
        return []

    if scene_list_file:
        logging.info(f"Loading scenes from file: {scene_list_file}")
        if not scene_list_file.is_file():
            logging.error(f"Scene list file not found: {scene_list_file}")
            return []
        
        with scene_list_file.open("r") as f:
            scene_names = [line.strip() for line in f if line.strip()]

        for scene_name in scene_names:
            scene_path = os.path.join(data_dir, scene_name)
            scenes.append(scene_path)
    else:
        logging.info("No scene list file provided, searching for all scene directories.")
        try:
            items = fs.ls(data_dir, detail=True)
        except Exception as e:
            logging.error(f"Failed to list directory {data_dir}: {e}")
            return []

        for item in items:
            if item["type"] == "directory":
                scene_path = item["name"]
                scenes.append(scene_path)

    logging.info(f"Found {len(scenes)} scenes.")
    return sorted(scenes)


def validate_sequence_files(out_path: str, fs, image_names: list[str] | None = None) -> tuple[bool, list[str]]:
    """
    Validate that all image and depth files in a sequence have non-zero sizes,
    and that all files listed in image_names actually exist.

    Returns:
        tuple: (is_valid, list_of_problematic_files)
    """
    problematic_files = []

    images_dir = os.path.join(out_path, IMAGE_FOLDER_NAME)
    depths_dir = os.path.join(out_path, DEPTH_FOLDER_NAME)

    if not fs.exists(images_dir) or not fs.exists(depths_dir):
        return False, ["Missing images or depths directory"]

    try:
        # List directories once and reuse results
        image_files = [f for f in fs.ls(images_dir, detail=True) if f["type"] == "file"]
        depth_files = [f for f in fs.ls(depths_dir, detail=True) if f["type"] == "file"]

        existing_images = {os.path.basename(f["name"]) for f in image_files}
        existing_depths = {os.path.basename(f["name"]) for f in depth_files}
        
        # Check that all files in image_names exist
        if image_names is not None:
            for img_name in image_names:
                if img_name not in existing_images:
                    problematic_files.append(f"Missing image: {img_name}")
                # Also check corresponding depth file
                depth_name = Path(img_name).stem + ".exr"
                if depth_name not in existing_depths:
                    problematic_files.append(f"Missing depth: {depth_name}")
        
        # Check file sizes for existing files (reuse already-fetched listings)
        for img_file in image_files:
            if img_file["size"] < MIN_FILE_SIZE_BYTES:
                name = os.path.basename(img_file["name"])
                problematic_files.append(f"Image {name}: {img_file['size']} bytes")

        for depth_file in depth_files:
            if depth_file["size"] < MIN_FILE_SIZE_BYTES:
                name = os.path.basename(depth_file["name"])
                problematic_files.append(f"Depth {name}: {depth_file['size']} bytes")
                
    except Exception as e:
        return False, [f"Error listing files: {e}"]

    return len(problematic_files) == 0, problematic_files


def process_scene(scene_path: str, fs, cfg: argparse.Namespace, data_dir: str):
    """
    Performs sanity checks on a single scene.
    """
    # Get relative path from data_dir instead of just basename
    scene_path_clean = scene_path.rstrip("/")
    data_dir_clean = data_dir.rstrip("/")
    if scene_path_clean.startswith(data_dir_clean):
        scene_name = scene_path_clean[len(data_dir_clean):].lstrip("/")
    else:
        scene_name = os.path.basename(scene_path_clean)

    # Load metadata first (needed for file validation)
    try:
        with fs.open(os.path.join(scene_path, "cam_from_worlds.npy"), "rb") as f:
            cam_from_worlds = np.load(BytesIO(f.read()))
        with fs.open(os.path.join(scene_path, "intrinsics.npy"), "rb") as f:
            intrinsics = np.load(BytesIO(f.read()))
        with fs.open(os.path.join(scene_path, "image_names.json"), "r") as f:
            image_names = json.load(f)
    except FileNotFoundError as e:
        reason = f"Missing metadata file: {e}"
        return scene_name, False, reason
    except Exception as e:
        reason = f"Error loading metadata: {e}"
        return scene_name, False, reason

    # 1. File existence and size check (now validates image_names entries exist)
    is_valid, problematic_files = validate_sequence_files(scene_path, fs, image_names)
    if not is_valid:
        reason = f"File validation failed: {problematic_files}"
        return scene_name, False, reason

    # 2. Metadata length consistency check
    num_extrinsics = len(cam_from_worlds)
    num_intrinsics = len(intrinsics)
    num_images = len(image_names)

    if not (num_extrinsics == num_intrinsics == num_images):
        reason = (
            "Mismatched metadata lengths: "
            f"cam_from_worlds={num_extrinsics}, "
            f"intrinsics={num_intrinsics}, "
            f"image_names={num_images}"
        )
        return scene_name, False, reason

    # 3. Extrinsic value check
    if np.any(np.abs(cam_from_worlds) > cfg.max_extrinsic_value):
        reason = f"Extrinsic values exceed threshold {cfg.max_extrinsic_value}"
        return scene_name, False, reason

    # All checks passed
    # logging.info(f"Scene {scene_name} passed sanity check.")
    return scene_name, True, None


def check_sanity_for_dataset(cfg):
    """
    Main function to find scenes and run sanity checks.
    """
    # Initialize filesystem
    fs, data_dir, is_s3 = get_fs(cfg.data_dir)
    logging.info(f"Using filesystem: {fs.protocol} (is_s3={is_s3})")
    
    scene_list_file = Path(cfg.scene_list_file) if cfg.scene_list_file else None
    all_scenes = find_scenes(data_dir, fs, scene_list_file)

    logging.info(f"Processing {len(all_scenes)} scenes...")

    successful_scenes = []
    failed_scenes = {}

    with tqdm(total=len(all_scenes), desc="Running Sanity Checks", unit="scene") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.num_workers) as executor:
            future_to_scene = {
                executor.submit(process_scene, scene_path, fs, cfg, data_dir): scene_path
                for scene_path in all_scenes
            }
            
            for future in concurrent.futures.as_completed(future_to_scene):
                scene_path = future_to_scene[future]
                try:
                    scene_name, success, error = future.result()
                    if not success:
                        failed_scenes[scene_name] = error
                        pbar.set_postfix_str(f"Failed: {scene_name}")
                    else:
                        successful_scenes.append(scene_name)
                        pbar.set_postfix_str(f"Completed: {scene_name}")
                except Exception as e:
                    logging.error(f"Unexpected error processing scene {scene_path}: {e}")
                    scene_name = os.path.basename(scene_path.rstrip("/"))
                    failed_scenes[scene_name] = str(e)
                pbar.update(1)

    # Write results to local files
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Use explicit dataset_name if provided, otherwise derive from data_dir
    if cfg.dataset_name:
        dataset_name = cfg.dataset_name
    else:
        dataset_name = os.path.basename(data_dir.rstrip("/"))
    success_file = os.path.join(output_dir, f"{dataset_name}_success.txt")
    failed_file = os.path.join(output_dir, f"{dataset_name}_failed.txt")

    with open(success_file, "w") as f:
        for name in sorted(successful_scenes):
            f.write(f"{name}\n")
            
    with open(failed_file, "w") as f:
        for name, reason in sorted(failed_scenes.items()):
            f.write(f"{name}: {reason}\n")

    logging.info(f"Sanity check complete.")
    logging.info(f"  Success: {len(successful_scenes)} (saved to {success_file})")
    logging.info(f"  Failed: {len(failed_scenes)} (saved to {failed_file})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform sanity checks on unified dataset scenes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the unified dataset directory (e.g., s3://dino/unified/apollo/)",
    )
    parser.add_argument(
        "--scene_list_file",
        type=str,
        default=None,
        help="Path to a txt file. Each row of the txt file will be a scene name.",
    )
    parser.add_argument(
        "--max_extrinsic_value",
        type=float,
        default=10000,
        help="Maximum absolute value allowed in the extrinsic matrix.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of threads to use for processing scenes.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/jianyuan/src/omega/projects/vggt/config/experiments/cw/cw_sanity_check/checks",
        help="Directory to save the success and failed scene lists.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Explicit dataset name for output files. If not provided, derived from data_dir.",
    )

    args = parser.parse_args()

    check_sanity_for_dataset(args)

