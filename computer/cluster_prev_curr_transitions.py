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
from multiprocessing import Pool

transform = transforms.ToTensor()

def load_image_pair(paths):
    prev_path, curr_path = paths
    prev_img = transform(Image.open(prev_path)).view(-1)
    curr_img = transform(Image.open(curr_path)).view(-1)
    return torch.cat([prev_img, curr_img], dim=0)

def compute_distance_matrix(df, device='cuda', num_workers=None):
    """Compute pairwise distance matrix between all images using GPU in one shot"""

    # Collect image paths
    image_paths = []
    for _, row in df.iterrows():
        record_num = row['record_num']
        image_num = row['image_num']
        if image_num == 0:
            prev_path = f'../data/data_processing/train_dataset/padding.png'
        else:
            prev_path = f'../data/data_processing/train_dataset/record_{record_num}/image_{image_num - 1}.png'
        curr_path = f'../data/data_processing/train_dataset/record_{record_num}/image_{image_num}.png'
        image_paths.append((prev_path, curr_path))
        assert os.path.exists(prev_path)
        assert os.path.exists(curr_path)

    print("Loading all images...")
    #if num_workers is None:
    #    num_workers = os.cpu_count() - 1
    #    num_workers = min(num_workers, 32)
    # Load images in parallel using multiprocessing
    #with Pool(num_workers) as pool:
    #    images = list(tqdm(
    #        pool.imap(load_image_pair, image_paths),
    #        total=len(image_paths)
    #    ))
    images_temp = []
    images = None
    for prev_path, curr_path in tqdm(image_paths):
        images_temp.append(load_image_pair((prev_path, curr_path)))
        if len(images_temp) == 10000:
            if images is None:
                images = torch.stack(images_temp).to(device)
            else:
                images = torch.cat([images, torch.stack(images_temp).to(device)], dim=0)
            images_temp = []
    if len(images_temp) > 0:
        if images is None:
            images = torch.stack(images_temp).to(device)
        else:
            images = torch.cat([images, torch.stack(images_temp).to(device)], dim=0)
    
    # Stack all images into a single tensor
    #images = torch.stack(images).to(device)
    
    print("Computing all pairwise distances...")
    with torch.no_grad():
        # Compute squared differences for all pairs at once
        # Using (a-b)^2 = a^2 + b^2 - 2ab formula
        norm = torch.norm(images,dim=1, keepdim=True) ** 2
        #a2 = torch.sum(images**2, dim=(1,2,3))[:, None]  # [N, 1]
        #b2 = torch.sum(images**2, dim=(1,2,3))[None, :]  # [1, N]
        #ab = torch.mm(images.view(images.size(0), -1), 
        #             images.view(images.size(0), -1).t())  # [N, N]
        distances = norm + norm.T - 2 * torch.einsum('ij,kj->ik', images, images)
        distances /= images.size(-1)
        #distances = (a2 + b2 - 2*ab) / (images.size(1) * images.size(2) * images.size(3))
        distances.clamp_(min=0)
        
        return distances.cpu().numpy()

def create_transition_image(record_num, image_num, history_length=3):
    """Create a horizontal strip of images showing the transition"""
    sequence_paths = []
    for i in range(history_length+1):
        if image_num - i < 0:
            sequence_paths.append(f'../data/data_processing/train_dataset/padding.png')
        else:
            sequence_paths.append(f'../data/data_processing/train_dataset/record_{record_num}/image_{image_num - i}.png')
    sequence_paths = sequence_paths[::-1]
    
    # Load all images
    images = [Image.open(path) for path in sequence_paths]
    
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
    # subsample df 
    df = df.sample(n=sample_size, random_state=random_seed)
    #df['Image_seq_cond_path'] = df['Image_seq_cond_path'].apply(ast.literal_eval)
    
    # Get unique target images and subsample if needed
    #target_images = df['Target_image'].unique()
    #print(f"Found {len(target_images)} unique target images")
    
    #if len(target_images) > sample_size:
    #    print(f"Subsampling to {sample_size} images...")
    #    np.random.seed(random_seed)
    #    target_images = np.random.choice(target_images, sample_size, replace=False)
    #    df = df[df['Target_image'].isin(target_images)].copy()
    
    # Compute distance matrix using GPU
    distances = compute_distance_matrix(df, device=device)
    
    # Run DBSCAN
    print("\nRunning DBSCAN clustering...")
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(distances)

    image_ids = list(range(len(df)))
    
    # Group images by cluster and sort by size
    clusters = defaultdict(list)
    for image_id, label in zip(image_ids, labels):
        clusters[label].append(image_id)
    
    # Convert to regular dict and sort by size (excluding noise cluster -1)
    sorted_clusters = {-1: clusters[-1]}  # Keep noise cluster first
    sorted_clusters.update({
        i: image_ids
        for i, (label, image_ids) in enumerate(
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
    for label, image_ids in sorted_clusters.items():
        if label == -1:
            print(f"Noise points: {len(image_ids)}")
        else:
            print(f"Cluster {label}: {len(image_ids)} images")
    
    # Save representative transitions
    print("\nSaving representative transitions...")
    for label, image_ids in sorted_clusters.items():
        if label == -1:
            cluster_dir = output_dir / "noise"
        else:
            cluster_dir = output_dir / f"cluster_{label:02d}"  # Zero-pad for nice sorting
        
        cluster_dir.mkdir(exist_ok=True)
        
        # Save cluster size in the directory name
        new_dir = cluster_dir.parent / f"{cluster_dir.name}_size_{len(image_ids)}"
        cluster_dir.rename(new_dir)
        cluster_dir = new_dir
        
        # Find sequences for this cluster's target images
        cluster_sequences = df.iloc[image_ids]
        
        # Compute and save cluster center (image with minimum average distance to other images)
        if len(image_ids) > 1 and label != -1:  # Don't compute for noise cluster
            # Calculate average distance from each point to all others in cluster
            cluster_distances = distances[np.ix_(image_ids, image_ids)]
            avg_distances = np.mean(cluster_distances, axis=1)
            center_idx = image_ids[np.argmin(avg_distances)]
            
            # Get the paths for the center image
            center_row = df.iloc[center_idx]
            record_num = center_row['record_num']
            image_num = center_row['image_num']
            if image_num == 0:
                prev_path = f'../data/data_processing/train_dataset/padding.png'
            else:
                prev_path = f'../data/data_processing/train_dataset/record_{record_num}/image_{image_num - 1}.png'
            curr_path = f'../data/data_processing/train_dataset/record_{record_num}/image_{image_num}.png'
            
            # Save cluster center
            if image_num == 0:
                center_image_prev = f'../data/data_processing/train_dataset/padding.png'
            else:
                center_image_prev = f'../data/data_processing/train_dataset/record_{record_num}/image_{image_num - 1}.png'
            center_image_curr = f'../data/data_processing/train_dataset/record_{record_num}/image_{image_num}.png'
            shutil.copy2(center_image_prev, cluster_dir / f"cluster_center_prev_{record_num}_{image_num-1}.png")
            shutil.copy2(center_image_curr, cluster_dir / f"cluster_center_curr_{record_num}_{image_num}.png")
        
        # Save representative transitions
        num_examples = min(5, len(cluster_sequences))
        for i, (_, row) in enumerate(cluster_sequences.sample(n=num_examples, random_state=random_seed).iterrows()):
            transition_image = create_transition_image(
                row['record_num'], 
                row['image_num'],
                history_length
            )
            transition_image.save(cluster_dir / f"transition_{i}.png")
    
    print(f"\nResults saved to {output_dir}")
    return sorted_clusters, distances, labels

if __name__ == "__main__":
    input_csv = "train_dataset/filtered_dataset.target_frames.csv"
    output_dir = "filtered_transition_clusters"
    
    # Parameters
    sample_size = 30000 # Number of images to sample
    #sample_size = 1000
    eps = 0.01  # Maximum distance between two samples to be in same cluster
    min_samples = 50  # Minimum number of samples in a cluster
    #min_samples = 1
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