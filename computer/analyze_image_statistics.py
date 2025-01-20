import pandas as pd
import numpy as np
from PIL import Image
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt

def analyze_image_statistics(csv_path):
    """Analyze pixel value statistics across the dataset"""
    df = pd.read_csv(csv_path)
    
    # Convert string representations to lists
    df['Image_seq_cond_path'] = df['Image_seq_cond_path'].apply(ast.literal_eval)
    
    # Initialize arrays for statistics
    means = []
    stds = []
    mins = []
    maxs = []
    
    print("Analyzing image statistics...")
    # Process both sequence images and target images
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Process sequence images
        for img_path in row['Image_seq_cond_path']:
            img = np.array(Image.open(img_path))
            means.append(img.mean())
            stds.append(img.std())
            mins.append(img.min())
            maxs.append(img.max())
        
        # Process target image
        target_img = np.array(Image.open(row['Target_image']))
        means.append(target_img.mean())
        stds.append(target_img.std())
        mins.append(target_img.min())
        maxs.append(target_img.max())
    
    # Convert to numpy arrays
    means = np.array(means)
    stds = np.array(stds)
    mins = np.array(mins)
    maxs = np.array(maxs)
    
    # Plot distributions
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    ax1.hist(means, bins=50)
    ax1.set_title('Distribution of Mean Pixel Values')
    ax1.set_xlabel('Mean Pixel Value')
    ax1.set_ylabel('Count')
    
    ax2.hist(stds, bins=50)
    ax2.set_title('Distribution of Pixel Standard Deviations')
    ax2.set_xlabel('Standard Deviation')
    ax2.set_ylabel('Count')
    
    ax3.hist(mins, bins=50)
    ax3.set_title('Distribution of Minimum Pixel Values')
    ax3.set_xlabel('Min Value')
    ax3.set_ylabel('Count')
    
    ax4.hist(maxs, bins=50)
    ax4.set_title('Distribution of Maximum Pixel Values')
    ax4.set_xlabel('Max Value')
    ax4.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('image_statistics.png')
    
    print("\nDataset Statistics:")
    print(f"Mean pixel value: {means.mean():.2f} ± {means.std():.2f}")
    print(f"Mean standard deviation: {stds.mean():.2f} ± {stds.std():.2f}")
    print(f"Overall min: {mins.min()}")
    print(f"Overall max: {maxs.max()}")
    
    return means.mean(), stds.mean(), mins.min(), maxs.max()

if __name__ == "__main__":
    csv_path = "train_dataset/train_dataset_2000.csv"
    stats = analyze_image_statistics(csv_path)
