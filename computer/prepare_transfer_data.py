import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm
import ast

def prepare_transfer_data(input_csv, output_dir, sample_size=2000, random_seed=42):
    """Prepare data for transfer by copying only necessary files"""
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
    
    # Get unique target images and subsample
    target_images = df['Target_image'].unique()
    print(f"Found {len(target_images)} unique target images")
    
    if len(target_images) > sample_size:
        print(f"Subsampling to {sample_size} images...")
        np.random.seed(random_seed)
        target_images = np.random.choice(target_images, sample_size, replace=False)
    
    # Collect all required image paths
    required_images = set()
    
    # Add sampled target images
    required_images.update(target_images)
    
    # Add corresponding sequence images for sampled targets
    #for _, row in df[df['Target_image'].isin(target_images)].iterrows():
    #    required_images.update(row['Image_seq_cond_path'])
    
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
        f.write(f"Sampled target images: {len(target_images)}\n")
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
    input_csv = "train_dataset/filtered_dataset.csv"
    output_dir = "transfer_data"
    sample_size = 20000
    
    prepare_transfer_data(input_csv, output_dir, sample_size)
