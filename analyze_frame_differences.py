import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import ast
import os
from tqdm import tqdm
import seaborn as sns

def compute_frame_difference(current_path, prev_path):
    """Compute MSE between two frames"""
    current_frame = np.array(Image.open(current_path))
    prev_frame = np.array(Image.open(prev_path))
    
    current_norm = current_frame.astype(float) / 255.0
    prev_norm = prev_frame.astype(float) / 255.0
    
    mse = np.mean((current_norm - prev_norm) ** 2)
    return mse

def analyze_dataset(csv_path, num_samples_per_quantile=5):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Convert string representations to lists
    df['Image_seq_cond_path'] = df['Image_seq_cond_path'].apply(ast.literal_eval)
    
    # Initialize lists to store differences and paths
    differences = []
    frame_pairs = []
    
    print("Computing frame differences...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        target_path = row['Target_image']
        prev_frame_path = row['Image_seq_cond_path'][-1]  # Last conditioning frame
        
        # Skip if either path contains 'padding.png'
        if 'padding.png' in prev_frame_path or 'padding.png' in target_path:
            continue
            
        diff = compute_frame_difference(target_path, prev_frame_path)
        differences.append(diff)
        frame_pairs.append((prev_frame_path, target_path))
    
    differences = np.array(differences)
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(differences, bins=50)
    plt.title('Distribution of Frame Differences (MSE)')
    plt.xlabel('Mean Squared Error')
    plt.ylabel('Count')
    plt.savefig('frame_differences_distribution.png')
    plt.close()
    
    # Create output directory
    os.makedirs('frame_difference_examples', exist_ok=True)
    
    # Sample from different quantiles
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantiles = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    quantiles = np.linspace(0.98, 1, num=50)
    
    print("\nSaving example frames for different quantiles...")
    for q in quantiles:
        threshold = np.quantile(differences, q)
        
        # Find indices close to this quantile
        indices = np.argsort(np.abs(differences - threshold))[:num_samples_per_quantile]
        
        for i, idx in enumerate(indices):
            prev_path, target_path = frame_pairs[idx]
            diff_value = differences[idx]
            
            # Create a side-by-side comparison
            prev_img = Image.open(prev_path)
            target_img = Image.open(target_path)
            
            # Create a new image with both frames side by side
            combined = Image.new('RGB', (prev_img.width * 2, prev_img.height))
            combined.paste(prev_img, (0, 0))
            combined.paste(target_img, (prev_img.width, 0))
            
            # Save with informative filename
            save_path = f'frame_difference_examples/quantile_{q:.2f}_sample_{i}_mse_{diff_value:.6f}.png'
            combined.save(save_path)
            
    print("\nAnalysis complete! Check:")
    print("- frame_differences_distribution.png for the distribution plot")
    print("- frame_difference_examples/ for sample frames at different quantiles")
    
    return differences, frame_pairs

if __name__ == "__main__":
    csv_path = "train_dataset/train_dataset_2000.csv"  # Update this path
    differences, frame_pairs = analyze_dataset(csv_path) 
