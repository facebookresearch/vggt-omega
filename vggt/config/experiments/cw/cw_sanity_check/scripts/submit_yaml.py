import sys
import os
import yaml
import subprocess
import argparse

def submit_job(yaml_file):
    if not os.path.exists(yaml_file):
        print(f"Error: {yaml_file} does not exist.")
        return

    try:
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading {yaml_file}: {e}")
        return

    unified_dir = config.get("UNIFIED_DIR")
    if not unified_dir:
        print(f"Skipping {yaml_file}: UNIFIED_DIR not found.")
        return

    sequence_list_file = config.get("sequence_list_file")
    
    dataset_name = os.path.splitext(os.path.basename(yaml_file))[0]
    job_name = f"check_{dataset_name}"
    
    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)
    
    output_log = f"outputs/{job_name}_%j.out"
    error_log = f"outputs/{job_name}_%j.err"
    
    # Path to submit_job.sh (assumed in same dir as this script)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "submit_job.sh")
    
    # Path to check_complete_aws.py (assumed in parent dir)
    parent_dir = os.path.dirname(current_dir)
    check_script_path = os.path.join(parent_dir, "check_complete_aws.py")
    
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--output={output_log}",
        f"--error={error_log}",
        f"--export=ALL,CHECK_SCRIPT={check_script_path}",
        script_path,
        f"--data_dir={unified_dir}",
        f"--dataset_name={dataset_name}",
        "--num_workers=96"
    ]
    
    if sequence_list_file:
        cmd.append(f"--scene_list_file={sequence_list_file}")
        
    print(f"Submitting job for {dataset_name}...")
    # print(" ".join(cmd))
    subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit a sanity check job from a dataset yaml config.")
    parser.add_argument("yaml_file", help="Path to the dataset yaml file")
    args = parser.parse_args()
    
    submit_job(args.yaml_file)

