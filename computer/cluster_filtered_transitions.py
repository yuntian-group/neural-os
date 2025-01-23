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

def create_transition_image(sequence_paths, target_path, history_length=3):
    """Create a horizontal strip of images showing the transition"""
    # Take last history_length images from sequence
    sequence_paths = sequence_paths[-history_length:]
    
    # Load all images
    images = [Image.open(path) for path in sequence_paths]
    images.append(Image.open(target_path))  # Add target image
    
    # Get dimensions
    width = images[0].width
    height = images[0].height
    
    # Create new image
    total_width = width * len(images)
    combined_image = Image.new('RGB', (total_width, height))
    
    # Paste images horizontally
    for i, img in enumerate(images):
        combined_image.paste(img, (i * width, 0))
    
    return combined_image

def cluster_transitions(input_csv, output_dir, sample_size=2000, eps=0.01, min_samples=50, 
                       random_seed=42, device='cuda', history_length=3):
    """Cluster target images and save representative transitions"""
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
    df['Image_seq_cond_path'] = df['Image_seq_cond_path'].apply(ast.literal_eval)
    
    # Get unique target images and subsample if needed
    target_images = df['Target_image'].unique()
    print(f"Found {len(target_images)} unique target images")
    
    if len(target_images) > sample_size:
        print(f"Subsampling to {sample_size} images...")
        np.random.seed(random_seed)
        target_images = np.random.choice(target_images, sample_size, replace=False)
        df = df[df['Target_image'].isin(target_images)].copy()
    
    # Compute distance matrix using GPU
    distances = compute_distance_matrix(target_images, device=device)
    
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
    
    # Save representative transitions
    print("\nSaving representative transitions...")
    for label, target_images in sorted_clusters.items():
        if label == -1:
            cluster_dir = output_dir / "noise"
        else:
            cluster_dir = output_dir / f"cluster_{label:02d}"  # Zero-pad for nice sorting
        
        cluster_dir.mkdir(exist_ok=True)
        
        # Save cluster size in the directory name
        new_dir = cluster_dir.parent / f"{cluster_dir.name}_size_{len(target_images)}"
        cluster_dir.rename(new_dir)
        cluster_dir = new_dir
        
        # Find sequences for this cluster's target images
        cluster_sequences = df[df['Target_image'].isin(target_images)]
        
        # Compute and save cluster center (image with minimum average distance to other images)
        if len(target_images) > 1 and label != -1:  # Don't compute for noise cluster
            indices = [list(target_images).index(img) for img in target_images]
            cluster_distances = distances[indices][:, indices]
            avg_distances = cluster_distances.mean(axis=1)
            center_idx = indices[np.argmin(avg_distances)]
            center_image = target_images[center_idx]
            
            # Save cluster center
            shutil.copy2(center_image, cluster_dir / "cluster_center.png")
        
        # Save representative transitions
        num_examples = min(5, len(cluster_sequences))
        for i, (_, row) in enumerate(cluster_sequences.sample(n=num_examples, random_state=random_seed).iterrows()):
            transition_image = create_transition_image(
                row['Image_seq_cond_path'], 
                row['Target_image'],
                history_length
            )
            transition_image.save(cluster_dir / f"transition_{i}.png")
    
    print(f"\nResults saved to {output_dir}")
    return sorted_clusters, distances, labels

if __name__ == "__main__":
    input_csv = "desktop_sequences.csv"
    output_dir = "desktop_transition_clusters"
    
    # Parameters
    sample_size = 30000 # Number of images to sample
    eps = 0.01  # Maximum distance between two samples to be in same cluster
    min_samples = 50  # Minimum number of samples in a cluster
    device = 'cuda'  # Use 'cpu' if no GPU available
    history_length = 3  # Number of previous frames to show in transitions
    
    clusters, distances, labels = cluster_transitions(
        input_csv, 
        output_dir, 
        sample_size=sample_size,
        eps=eps, 
        min_samples=min_samples,
        device=device,
        history_length=history_length
    )
