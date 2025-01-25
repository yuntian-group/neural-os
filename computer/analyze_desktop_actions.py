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
from collections import defaultdict

# Constants
ICONS = {
    'firefox': {'center': (66, 332-20), 'radius': int(22*1.2)},
    'root': {'center': (66, 185-20), 'radius': int(22*1.2)},
    'terminal': {'center': (191, 60-20), 'radius': int(22*1.2)},
    'trash': {'center': (66, 60-20), 'radius': int(22*1.2)}
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
    # Remove all spaces
    action_str = action_str.replace(" ", "")
    
    # First character is action type, rest is coordinates
    action_type, action_str = action_str[0], action_str[1:]
    
    # Split by ':' and convert to integers
    coords = action_str.split(':')
    coords = [int(item) for item in coords]
    
    # Assert we have exactly two coordinates
    assert len(coords) == 2, f"Expected 2 coordinates, got {len(coords)} from action string: {action_str}"
    
    return action_type, (coords[0], coords[1])

def is_double_click(actions, time_threshold=0.3):
    """Detect double click by finding two clicks close in time/space, focusing on the last valid occurrence"""
    click_actions = [(i, parse_action(a)) for i, a in enumerate(actions) if parse_action(a)[0] == 'L']
    
    # Traverse clicks in reverse to find the last valid double click
    for i in range(len(click_actions)-1, 0, -1):  # Start from the end
        curr_idx, (_, curr_pos) = click_actions[i]
        prev_idx, (_, prev_pos) = click_actions[i-1]
        
        # Check if clicks are close in time (frames) and space
        time_diff = (curr_idx - prev_idx) * 0.1  # 10 fps -> 0.1s per frame
        dist = np.sqrt(sum((a-b)**2 for a, b in zip(curr_pos, prev_pos)))
        
        if time_diff <= time_threshold and dist < 10:  # 10 pixels threshold
            # Check if this double click is on any icon
            for name, icon in ICONS.items():
                icon_dist = np.sqrt(sum((a-b)**2 for a, b in zip(curr_pos, icon['center'])))
                if icon_dist <= icon['radius']:
                    return True, curr_pos  # Return position of the second (later) click
    
    return False, None

def predict_target(action_sequence):
    """Predict target based on action sequence"""
    # Check for double click
    has_double_click, click_pos = is_double_click(action_sequence)
    #import pdb; pdb.set_trace()
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
    
    # Only return if we found a valid icon
    if closest_icon is None:
        return None
        
    return closest_icon

def visualize_sequence(image_paths, target_image, action_sequence, save_path, history_length=14):
    """Visualize action sequence with cursor positions and clicks"""
    images = []
    
    # Take last history_length frames and corresponding actions
    frame_paths = image_paths[-history_length:]
    total_frames = len(frame_paths)
    
    # Process each frame with its corresponding action
    for i, img_path in enumerate(frame_paths):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw icon centers and square boundaries
        for name, icon in ICONS.items():
            center = icon['center']
            radius = icon['radius']
            # Draw center point
            cv2.circle(img, center, 2, (0, 255, 0), -1)  # Green dot for center
            # Draw square boundary
            x1 = center[0] - radius
            y1 = center[1] - radius
            x2 = center[0] + radius
            y2 = center[1] + radius
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Yellow square for boundary
        
        # Get corresponding action for this frame
        action = action_sequence[-len(frame_paths) + i]
        action_type, (x, y) = parse_action(action)
        
        # Draw cursor position
        cv2.circle(img, (x, y), 3, (255, 255, 255), -1)
        
        # Draw click if present
        if action_type == 'L':
            cv2.circle(img, (x, y), 10, (255, 0, 0), 3)  # Thicker blue circle for clicks
        
        images.append(img)
    
    # Add target image (without icon visualizations)
    target_img = cv2.imread(target_image)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    images.append(target_img)
    
    # Create 3x5 grid for 14 frames + target
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    # Plot all images
    for i in range(15):
        row = i // 5
        col = i % 5
        axes[row, col].imshow(images[i])
        axes[row, col].axis('off')
        if i < 14:
            axes[row, col].set_title(f'Frame {i+1}')
        else:
            axes[row, col].set_title('Target')
    
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
    error_cases = defaultdict(list)
    
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
        
        # Save error cases grouped by confusion type
        if prediction != ground_truth:
            error_key = f"{ground_truth}_{prediction if prediction else 'None'}"
            error_cases[error_key].append({
                'idx': idx,
                'image_seq': image_seq,
                'action_seq': action_seq,
                'target_image': target_image
            })
    
    # Compute and print metrics
    results_df = pd.DataFrame(results)
    accuracy = results_df['correct'].mean()
    
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Total examples: {len(results_df)}")
    
    print("\nConfusion Matrix:")
    # Print confusion matrix in a clearer format
    for true_class in ICONS.keys():
        true_cases = results_df[results_df['ground_truth'] == true_class]
        if len(true_cases) == 0:
            print(f"\nTrue {true_class}: No examples found")
        else:
            print(f"\nTrue {true_class}: ({len(true_cases)} examples)")
            predictions = true_cases['prediction'].value_counts(dropna=False)
            for pred_class, count in predictions.items():
                if pd.isna(pred_class):
                    print(f"  No prediction: {count}/{len(true_cases)} ({count/len(true_cases):.1%})")
                else:
                    print(f"  Predicted {pred_class}: {count}/{len(true_cases)} ({count/len(true_cases):.1%})")
    
    print("\nPrediction distribution:")
    print(results_df['prediction'].value_counts(dropna=False))
    
    # Save error cases by confusion type
    for error_key, cases in error_cases.items():
        if cases:  # Only create directories for error types that exist
            error_dir = output_dir / error_key
            error_dir.mkdir(exist_ok=True)
            
            for i, case in enumerate(cases):
                visualize_sequence(
                    case['image_seq'],
                    case['target_image'],
                    case['action_seq'],
                    error_dir / f"sequence_{i}.png",
                    history_length=history_length
                )
    
    return results_df, error_cases

if __name__ == "__main__":
    csv_path = "desktop_sequences_filtered.csv"
    output_dir = "desktop_analysis_results"
    history_length = 14  # Number of previous frames to show in transitions
    debug = True # Set to True to process only first 100 rows
    
    results_df, error_cases = analyze_sequences(
        csv_path, 
        output_dir,
        debug=debug,
        history_length=history_length
    )
