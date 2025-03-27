import os
import re
import shutil
from collections import defaultdict

def cleanup_folders(base_dir, dry_run=True):
    folder_pattern = re.compile(r"^(.*)_ckpt(\d+)$")
    folders = defaultdict(list)

    for folder in os.listdir(base_dir):
        if 'psearch_a_vis_norm_standard_contexta' not in folder:
            continue
        match = folder_pattern.match(folder)
        if match:
            prefix, ckpt_num = match.group(1), int(match.group(2))
            folders[prefix].append((ckpt_num, folder))

    for prefix, ckpt_list in folders.items():
        ckpt_list.sort(reverse=True)  # Sort descending by checkpoint number
        latest_ckpt = ckpt_list[0][1]
        older_ckpts = [folder for _, folder in ckpt_list[1:]]

        print(f"\nPrefix: '{prefix}'")
        print(f"Keeping latest checkpoint: '{latest_ckpt}'")
        if older_ckpts:
            print(f"Removing older checkpoints: {older_ckpts}")
            for folder in older_ckpts:
                folder_path = os.path.join(base_dir, folder)
                if dry_run:
                    print(f"[Dry run] Would remove: {folder_path}")
                else:
                    shutil.rmtree(folder_path)
                    print(f"Removed: {folder_path}")
        else:
            print("No older checkpoints to remove.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean up old ckpt folders")
    parser.add_argument("--dir", type=str, default=".", help="Base directory containing folders")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without deleting folders")

    args = parser.parse_args()

    cleanup_folders(args.dir, dry_run=args.dry_run)

