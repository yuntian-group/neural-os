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

def compute_distance_matrix(image_paths, device='cuda'):
    """Compute pairwise distance matrix between all images using GPU in one shot"""
    transform = transforms.ToTensor()
    
    print("Loading all images...")
    # Load all images into a single tensor [N, C, H, W]
    images = torch.stack([
        transform(Image.open(path)) 
        for path in tqdm(image_paths)
    ]).to(device)
    
    print("Computing all pairwise distances...")
    with torch.no_grad():
        # Compute squared differences for all pairs at once
        # Using (a-b)^2 = a^2 + b^2 - 2ab formula
        a2 = torch.sum(images**2, dim=(1,2,3))[:, None]  # [N, 1]
        b2 = torch.sum(images**2, dim=(1,2,3))[None, :]  # [1, N]
        ab = torch.mm(images.view(images.size(0), -1), 
                     images.view(images.size(0), -1).t())  # [N, N]
        distances = (a2 + b2 - 2*ab) / (images.size(1) * images.size(2) * images.size(3))
        distances = torch.clamp(distances, min=0)
        
        return distances.cpu().numpy()

def cluster_images(input_csv, output_dir, sample_size=2000, eps=0.1, min_samples=5, 
                  random_seed=42, device='cuda'):
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
    distances = compute_distance_matrix(target_images, device=device)
    import pdb; pdb.set_trace()
    
    # Run DBSCAN
    print("\nRunning DBSCAN clustering...")
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(distances)
    
    # Group images by cluster and sort by size
    clusters = defaultdict(list)
    for img_path, label in zip(target_images, labels):
        clusters[label].append(img_path)
    
    # Convert to regular dict and sort by size (excluding noise cluster -1)
    sorted_clusters = {-1: clusters[-1]}  # Keep noise cluster first
    sorted_clusters.update({
        i: cluster_images
        for i, (label, cluster_images) in enumerate(
            sorted(
                [(k, v) for k, v in clusters.items() if k != -1],
                key=lambda x: len(x[1]),
                reverse=True  # Largest clusters first
            )
        )
    })
    
    # Print clustering results
    print("\nClustering Results:")
    print(f"Number of clusters found: {len(sorted_clusters)-1}")  # -1 because -1 is noise
    print("\nCluster sizes:")
    for label, images in sorted_clusters.items():
        if label == -1:
            print(f"Noise points: {len(images)}")
        else:
            print(f"Cluster {label}: {len(images)} images")
    
    # Save representative examples
    print("\nSaving representative examples...")
    for label, images in sorted_clusters.items():
        if label == -1:
            cluster_dir = output_dir / "noise"
        else:
            cluster_dir = output_dir / f"cluster_{label:02d}"  # Zero-pad for nice sorting
        
        cluster_dir.mkdir(exist_ok=True)
        
        # Save cluster size in the directory name
        new_dir = cluster_dir.parent / f"{cluster_dir.name}_size_{len(images)}"
        cluster_dir.rename(new_dir)
        cluster_dir = new_dir
        
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
    return sorted_clusters, distances, labels

if __name__ == "__main__":
    input_csv = "train_dataset/filtered_dataset.csv"
    output_dir = "clustering_results"
    
    # Parameters
    sample_size = 20000  # Number of images to sample
    eps = 0.01  # Maximum distance between two samples to be in same cluster
    min_samples = 50  # Minimum number of samples in a cluster
    device = 'cuda'  # Use 'cpu' if no GPU available
    #device = 'cpu'  # Use 'cpu' if no GPU available
    
    clusters, distances, labels = cluster_images(
        input_csv, 
        output_dir, 
        sample_size=sample_size,
        eps=eps, 
        min_samples=min_samples,
        device=device
    )
