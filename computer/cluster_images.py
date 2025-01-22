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

def compute_frame_difference(img1_path, img2_path):
    """Compute MSE between two frames"""
    img1 = np.array(Image.open(img1_path))
    img2 = np.array(Image.open(img2_path))
    
    img1_norm = img1.astype(float) / 255.0
    img2_norm = img2.astype(float) / 255.0
    
    mse = np.mean((img1_norm - img2_norm) ** 2)
    return mse

def compute_distance_matrix(image_paths, num_workers=None):
    """Compute pairwise distance matrix between all images"""
    n = len(image_paths)
    distances = np.zeros((n, n))
    
    print("Computing pairwise distances...")
    for i in tqdm(range(n)):
        for j in range(i+1, n):
            dist = compute_frame_difference(image_paths[i], image_paths[j])
            distances[i,j] = dist
            distances[j,i] = dist
    
    return distances

def cluster_images(input_csv, output_dir, eps=0.1, min_samples=5):
    """Cluster images and save representatives"""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read dataset
    print("Reading dataset...")
    df = pd.read_csv(input_csv)
    
    # Get unique target images
    target_images = df['Target_image'].unique()
    print(f"Found {len(target_images)} unique target images")
    
    # Compute distance matrix
    distances = compute_distance_matrix(target_images)
    
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
    
    # You might need to tune these parameters
    eps = 0.1  # Maximum distance between two samples to be in same cluster
    min_samples = 5  # Minimum number of samples in a cluster
    
    clusters, distances, labels = cluster_images(input_csv, output_dir, eps, min_samples)
