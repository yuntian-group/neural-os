import pandas as pd
import numpy as np
from PIL import Image
import ast
import cv2
from pathlib import Path
import torch
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import re

# Constants
ICONS = {
    'firefox': {'center': (66, 332), 'radius': 22},
    'root': {'center': (66, 185), 'radius': 22},
    'terminal': {'center': (191, 60), 'radius': 22},
    'trash': {'center': (66, 60), 'radius': 22}
}

CLUSTER_PATHS = {
    'terminal': "desktop_transition_clusters/cluster_01_size_1499_desktop_terminal/cluster_center.png",
    'firefox': "desktop_transition_clusters/cluster_03_size_1275_desktop_firefox/cluster_center.png",
    'root': "desktop_transition_clusters/cluster_04_size_799_desktop_root/cluster_center.png",
    'trash': "desktop_transition_clusters/cluster_05_size_738_desktop_trash/cluster_center.png"
}

def compute_frame_difference(img1_path, img2_path, device='cpu'):
    """Compute MSE between two images"""
    transform = transforms.ToTensor()
    
    img1 = transform(Image.open(img1_path)).to(device)
    img2 = transform(Image.open(img2_path)).to(device)
    
    with torch.no_grad():
        distance = torch.mean((img2 - img1) ** 2)
        return float(distance.cpu())

def get_ground_truth(target_image):
    """Determine ground truth by comparing with cluster centers"""
    distances = {}
    for name, cluster_path in CLUSTER_PATHS.items():
        distances[name] = compute_frame_difference(target_image, cluster_path)
    return min(distances.items(), key=lambda x: x[1])[0]

def parse_action(action_str):
    """Parse action string into type and coordinates"""
    parts = action_str.split('+')
    action_type = parts[0].strip()
    coords = [int(x) for x in re.findall(r'\d+', parts[1])]
    return action_type, (coords[0], coords[1])

def is_double_click(actions, time_threshold=0.3):
    """Detect double click by finding two clicks close in time/space, focusing on the last occurrence"""
    click_actions = [(i, parse_action(a)) for i, a in enumerate(actions) if parse_action(a)[0] == 'L']
    
    # Traverse clicks in reverse to find the last double click
    for i in range(len(click_actions)-1, 0, -1):  # Start from the end
        curr_idx, (_, curr_pos) = click_actions[i]
        prev_idx, (_, prev_pos) = click_actions[i-1]
        
        # Check if clicks are close in time (frames) and space
        time_diff = (curr_idx - prev_idx) * 0.1  # 10 fps -> 0.1s per frame
        dist = np.sqrt(sum((a-b)**2 for a, b in zip(curr_pos, prev_pos)))
        
        if time_diff <= time_threshold and dist < 10:  # 10 pixels threshold
            return True, curr_pos  # Return position of the second (later) click
    
    return False, None

def predict_target(action_sequence):
    """Predict target based on action sequence"""
    # Check for double click
    has_double_click, click_pos = is_double_click(action_sequence)
    if not has_double_click:
        return None
    
    # Find closest icon to click position
    min_dist = float('inf')
    closest_icon = None
    
    for name, icon in ICONS.items():
        dist = np.sqrt(sum((a-b)**2 for a, b in zip(click_pos, icon['center'])))
        if dist < min_dist and dist <= icon['radius']:
            min_dist = dist
            closest_icon = name
    
    return closest_icon

def visualize_sequence(image_paths, action_sequence, save_path, history_length=7):
    """Visualize action sequence with cursor positions and clicks"""
    images = []
    for img_path in image_paths[-history_length:]:  # Last N frames
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw all cursor positions and clicks
        for action in action_sequence:
            action_type, (x, y) = parse_action(action)
            
            # Draw cursor position
            cv2.circle(img, (x, y), 3, (255, 255, 255), -1)
            
            # Draw click if present
            if action_type == 'L':
                cv2.circle(img, (x, y), 10, (255, 0, 0), 2)
        
        images.append(img)
    
    # Create horizontal strip
    fig, axes = plt.subplots(1, len(images), figsize=(5*len(images), 5))
    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Frame {i+1}')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_sequences(csv_path, output_dir="analysis_results", debug=False, history_length=7):
    """Analyze all sequences and compute accuracy"""
    df = pd.read_csv(csv_path)
    
    if debug:
        print("Debug mode: using first 100 rows only")
        df = df.head(100)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    results = []
    error_cases = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_seq = ast.literal_eval(row['Image_seq_cond_path'])
        action_seq = ast.literal_eval(row['Action_seq'])
        target_image = row['Target_image']
        
        # Get prediction and ground truth
        prediction = predict_target(action_seq)
        ground_truth = get_ground_truth(target_image)
        
        results.append({
            'idx': idx,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'correct': prediction == ground_truth
        })
        
        # Save error cases
        if prediction != ground_truth:
            error_case_dir = output_dir / f"error_{idx}"
            error_case_dir.mkdir(exist_ok=True)
            
            visualize_sequence(
                image_seq,
                action_seq,
                error_case_dir / "sequence.png",
                history_length=history_length
            )
            
            error_cases.append({
                'idx': idx,
                'prediction': prediction,
                'ground_truth': ground_truth,
                'image_seq': image_seq,
                'action_seq': action_seq
            })
    
    # Compute and print metrics
    results_df = pd.DataFrame(results)
    accuracy = results_df['correct'].mean()
    
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print("\nConfusion Matrix:")
    for true_class in ICONS.keys():
        true_cases = results_df[results_df['ground_truth'] == true_class]
        print(f"\nTrue {true_class}:")
        print(true_cases['prediction'].value_counts(normalize=True))
    
    # Save error cases summary
    pd.DataFrame(error_cases).to_csv(output_dir / "error_cases.csv", index=False)
    
    return results_df, error_cases

if __name__ == "__main__":
    csv_path = "desktop_sequences_filtered.csv"
    output_dir = "desktop_analysis_results"
    history_length = 7  # Number of previous frames to show in transitions
    debug = False  # Set to True to process only first 100 rows
    
    results_df, error_cases = analyze_sequences(
        csv_path, 
        output_dir,
        debug=debug,
        history_length=history_length
    )
