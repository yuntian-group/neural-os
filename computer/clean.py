#!/usr/bin/env python3
import os
import re
import time
import argparse

# Regular expression to extract the numeric step from filenames.
CHECKPOINT_REGEX = re.compile(r"model-step=(\d+)\.ckpt")

def get_checkpoint_step(filename):
    """
    If the filename matches the checkpoint pattern,
    return the integer step; otherwise, return None.
    """
    match = CHECKPOINT_REGEX.search(filename)
    if match:
        return int(match.group(1))
    return None

def clean_folder(folder):
    """
    In the given folder, remove any checkpoint file whose generation
    (defined as step // 100000) is lower than the highest generation present.
    """
    try:
        filenames = os.listdir(folder)
    except Exception as e:
        print(f"Error listing folder {folder}: {e}")
        return

    # Gather checkpoint files and their step values.
    checkpoints = {}
    for fname in filenames:
        step = get_checkpoint_step(fname)
        if step is not None:
            checkpoints[fname] = step

    if not checkpoints:
        # No checkpoint files found in this folder.
        return

    # Determine the generation based on the highest checkpoint available.
    max_step = max(checkpoints.values())
    current_generation = max_step // 1000
    print(f"[{folder}] Max checkpoint step: {max_step} (Generation {current_generation})")

    # Remove checkpoints from older generations.
    for fname, step in checkpoints.items():
        file_generation = step // 1000
        if file_generation < current_generation:
            full_path = os.path.join(folder, fname)
            try:
                os.remove(full_path)
                print(f"Removed older checkpoint: {full_path} (Generation {file_generation})")
            except Exception as e:
                print(f"Error removing file {full_path}: {e}")

def monitor_folders(folders, interval=10):
    """
    Monitor each folder in the list. Every `interval` seconds,
    check each folder and clean older checkpoints.
    """
    print("Starting checkpoint monitor. Press Ctrl+C to exit.")
    try:
        while True:
            for folder in folders:
                clean_folder(folder)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")

def find_checkpoint_folders(substring="saved_standard_challenging_context", base_dir="."):
    """
    Search the given base directory for folders whose names contain the
    specified substring.
    """
    folders = []
    try:
        for entry in os.listdir(base_dir):
            full_path = os.path.join(base_dir, entry)
            if substring in entry and os.path.isdir(full_path):
                folders.append(full_path)
    except Exception as e:
        print(f"Error accessing base directory {base_dir}: {e}")
    return folders

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Monitor folders whose names contain 'saved_standard_challenging_context' "
                     "and remove older checkpoints when newer ones are available.")
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Polling interval in seconds (default: 10)"
    )
    parser.add_argument(
        "--base_dir",
        default="./train_dataset_encoded5/",
        help="Base directory to search for folders (default: current directory)"
    )
    parser.add_argument(
        "--substring",
        default="saved_",
        help="Substring to search for in folder names (default: 'saved_standard_challenging_context')"
    )
    args = parser.parse_args()

    # Automatically find all folders in base_dir that contain the specified substring.
    folders = find_checkpoint_folders(substring=args.substring, base_dir=args.base_dir)
    if not folders:
        print(f"No folders found in '{args.base_dir}' containing '{args.substring}'.")
    else:
        print("Monitoring the following folders:")
        for folder in folders:
            print(" -", folder)
        monitor_folders(folders, args.interval)

