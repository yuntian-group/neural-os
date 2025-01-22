import pandas as pd
import numpy as np
from PIL import Image
import ast
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from collections import defaultdict
import os
import shutil
from pathlib import Path
import torch
from torchvision import transforms

def compute_frame_difference_batch(img1_paths, img2_paths, batch_size=100, device='cuda'):
    """Compute MSE between batches of image pairs using GPU"""
    transform = transforms.Compose([
        transforms.ToTensor(),  # This also scales to [0, 1]
    ])
    
    n = len(img1_paths)
    distances = []
    
    for i in tqdm(range(0, n, batch_size)):
        batch_end = min(i + batch_size, n)
        
        # Load and process first set of images
        imgs1 = torch.stack([
            transform(Image.open(path)) 
            for path in img1_paths[i:batch_end]
        ]).to(device)
        
        # Load and process second set of images
        imgs2 = torch.stack([
            transform(Image.open(path))
            for path in img2_paths[i:batch_end]
        ]).to(device)
        
        # Compute MSE
        with torch.no_grad():
            batch_distances = torch.mean((imgs1 - imgs2) ** 2, dim=(1, 2, 3))
            distances.extend(batch_distances.cpu().numpy())
    
    return np.array(distances)

def compute_distance_matrix(image_paths, batch_size=100, device='cuda'):
    """Compute pairwise distance matrix between all images using GPU"""
    n = len(image_paths)
    distances = np.zeros((n, n))
    
    print("Computing pairwise distances...")
    for i in tqdm(range(n)):
        # Create pairs for current row
        img1_paths = [image_paths[i]] * (n - i - 1)
        img2_paths = image_paths[i+1:]
        
        if len(img2_paths) > 0:
            # Compute distances for current row
            row_distances = compute_frame_difference_batch(
                img1_paths, 
                img2_paths, 
                batch_size=batch_size,
                device=device
            )
            
            # Fill in the matrix (symmetric)
            distances[i, i+1:] = row_distances
            distances[i+1:, i] = row_distances
    
    return distances

def cluster_images(input_csv, output_dir, sample_size=2000, eps=0.1, min_samples=5, 
                  random_seed=42, batch_size=100, device='cuda'):
    """Cluster images and save representatives"""
    # Check if CUDA is available when device is 'cuda'
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read dataset
    print("Reading dataset...")
    df = pd.read_csv(input_csv)
    
    # Get unique target images and subsample
    target_images = df['Target_image'].unique()
    print(f"Found {len(target_images)} unique target images")
    
    if len(target_images) > sample_size:
        print(f"Subsampling to {sample_size} images...")
        np.random.seed(random_seed)
        target_images = np.random.choice(target_images, sample_size, replace=False)
    
    # Compute distance matrix using GPU
    distances = compute_distance_matrix(target_images, batch_size=batch_size, device=device)
    
    # Run DBSCAN
    print("\nRunning DBSCAN clustering...")
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(distances)
    
    # Group images by cluster
    clusters = defaultdict(list)
    for img_path, label in zip(target_images, labels):
        clusters[label].append(img_path)
    
    # Print clustering results
    print("\nClustering Results:")
    print(f"Number of clusters found: {len(clusters)-1}")  # -1 because -1 is noise
    print("\nCluster sizes:")
    for label, images in sorted(clusters.items()):
        if label == -1:
            print(f"Noise points: {len(images)}")
        else:
            print(f"Cluster {label}: {len(images)} images")
    
    # Save representative examples
    print("\nSaving representative examples...")
    for label, images in clusters.items():
        if label == -1:
            cluster_dir = output_dir / "noise"
        else:
            cluster_dir = output_dir / f"cluster_{label}"
        
        cluster_dir.mkdir(exist_ok=True)
        
        # Compute cluster center (image with minimum average distance to other images in cluster)
        if len(images) > 1:
            indices = [list(target_images).index(img) for img in images]
            cluster_distances = distances[indices][:, indices]
            avg_distances = cluster_distances.mean(axis=1)
            center_idx = indices[np.argmin(avg_distances)]
            center_image = target_images[center_idx]
            
            # Save cluster center
            shutil.copy2(center_image, cluster_dir / "cluster_center.png")
            
            # Save a few random examples (if available)
            num_examples = min(5, len(images))
            for i, img_path in enumerate(np.random.choice(images, num_examples, replace=False)):
                shutil.copy2(img_path, cluster_dir / f"example_{i}.png")
        else:
            # If only one image in cluster/noise, just save it
            shutil.copy2(images[0], cluster_dir / "single_example.png")
    
    print(f"\nResults saved to {output_dir}")
    return clusters, distances, labels

if __name__ == "__main__":
    input_csv = "train_dataset/filtered_dataset.csv"
    output_dir = "clustering_results"
    
    # Parameters
    sample_size = 2000  # Number of images to sample
    eps = 0.1  # Maximum distance between two samples to be in same cluster
    min_samples = 5  # Minimum number of samples in a cluster
    batch_size = 100  # Number of image pairs to process in parallel
    device = 'cuda'  # Use 'cpu' if no GPU available
    
    clusters, distances, labels = cluster_images(
        input_csv, 
        output_dir, 
        sample_size=sample_size,
        eps=eps, 
        min_samples=min_samples,
        batch_size=batch_size,
        device=device
    )
