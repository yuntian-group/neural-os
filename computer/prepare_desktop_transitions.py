import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
import ast

def prepare_desktop_data(input_csv, output_dir, history_length=3):
    """Prepare desktop transition data by copying all necessary files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directories
    data_dir = output_dir / "data"
    csv_dir = data_dir / "csv"
    image_dir = data_dir / "images"
    
    for dir_path in [data_dir, csv_dir, image_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Copy CSV file
    print(f"Copying CSV file: {input_csv}")
    shutil.copy2(input_csv, csv_dir)
    
    # Read dataset and get unique images
    print("Reading dataset and collecting image paths...")
    df = pd.read_csv(input_csv)
    df['Image_seq_cond_path'] = df['Image_seq_cond_path'].apply(ast.literal_eval)
    
    # Collect all required image paths
    required_images = set()
    
    # Add all target images
    required_images.update(df['Target_image'])
    
    # Add all sequence images
    for seq in df['Image_seq_cond_path']:
        required_images.update(seq[-history_length:])
    
    # Copy images while maintaining directory structure
    print(f"\nCopying {len(required_images)} unique images...")
    for src_path in tqdm(required_images):
        src_path = Path(src_path)
        rel_path = src_path.relative_to(Path(input_csv).parent)
        dst_path = image_dir / rel_path
        
        # Create parent directories if they don't exist
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the file
        shutil.copy2(src_path, dst_path)
    
    # Create a manifest file
    print("\nCreating manifest file...")
    with open(output_dir / "manifest.txt", "w") as f:
        f.write(f"Total unique images: {len(required_images)}\n")
        f.write(f"Total sequences: {len(df)}\n")
        f.write(f"Original CSV: {input_csv}\n")
        f.write("\nDirectory structure:\n")
        f.write(f"- {output_dir.name}/\n")
        f.write("  - data/\n")
        f.write("    - csv/\n")
        f.write("    - images/\n")
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file())
    print(f"\nPreparation complete!")
    print(f"Total data size: {total_size / (1024*1024*1024):.2f} GB")
    print(f"Data prepared in: {output_dir}")
    
    return output_dir

if __name__ == "__main__":
    input_csv = "desktop_sequences.csv"
    output_dir = "desktop_data"
    
    prepare_desktop_data(input_csv, output_dir)
