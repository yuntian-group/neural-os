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
import pickle
EPS = 1e-5
# Constants
ICONS = {
    'firefox': {'center': (66, 332-30), 'width': int(22*1.4), 'height': 44},
    'root': {'center': (66, 185), 'width': int(22*1.95), 'height': 42},
    'terminal': {'center': (191, 60), 'width': int(22*2), 'height': 44},
    'trash': {'center': (66, 60), 'width': int(22*1.95), 'height': 42}
}

CLUSTER_PATHS = {
    'desktop': "desktop_transition_clusters/cluster_00_size_24373_desktop_desktop/cluster_center.png",
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
        
        # Check if clicks are close in time
        time_diff = (curr_idx - prev_idx) * 0.1  # 10 fps -> 0.1s per frame
        
        if time_diff <= time_threshold + EPS:
            # Check if both clicks are on the same icon
            for name, icon in ICONS.items():
                center = icon['center']
                width = icon['width']
                height = icon['height']
                # Check if both clicks are within this icon's boundary
                curr_in_icon = (abs(curr_pos[0] - center[0]) <= width and 
                              abs(curr_pos[1] - center[1]) <= height)
                prev_in_icon = (abs(prev_pos[0] - center[0]) <= width and 
                              abs(prev_pos[1] - center[1]) <= height)
                
                if curr_in_icon and prev_in_icon:
                    return True, curr_pos  # Return position of the second (later) click
    
    return False, None

def predict_target(action_sequence, time_threshold):
    """Predict target based on action sequence"""
    # Check for double click
    has_double_click, click_pos = is_double_click(action_sequence, time_threshold)
    if not has_double_click:
        return 'desktop'
    
    # Find closest icon to click position
    min_dist = float('inf')
    closest_icon = 'desktop'
    
    for name, icon in ICONS.items():
        # Check if click is within rectangular boundary
        if (abs(click_pos[0] - icon['center'][0]) <= icon['width'] and 
            abs(click_pos[1] - icon['center'][1]) <= icon['height']):
            dist = max(abs(click_pos[0] - icon['center'][0]) / icon['width'], 
                      abs(click_pos[1] - icon['center'][1]) / icon['height'])
            if dist < min_dist:
                min_dist = dist
                closest_icon = name
    
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
        
        # Draw icon centers and rectangular boundaries
        for name, icon in ICONS.items():
            center = icon['center']
            width = icon['width']
            height = icon['height']
            # Draw center point
            cv2.circle(img, center, 2, (0, 255, 0), -1)  # Green dot for center
            # Draw rectangular boundary
            x1 = center[0] - width
            y1 = center[1] - height
            x2 = center[0] + width
            y2 = center[1] + height
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Yellow rectangle for boundary
        
        # Get corresponding action for this frame
        action = action_sequence[-len(frame_paths) + i]
        action_type, (x, y) = parse_action(action)
        
        # Draw cursor position and clicks
        cv2.circle(img, (x, y), 3, (255, 255, 255), -1)
        if action_type == 'L':
            # Check if click is inside any icon boundary
            inside_icon = False
            for name, icon in ICONS.items():
                center = icon['center']
                width = icon['width']
                height = icon['height']
                if (abs(x - center[0]) <= width and 
                    abs(y - center[1]) <= height):
                    inside_icon = True
                    break
            
            # Green circle for clicks inside icons, red for clicks outside
            color = (0, 255, 0) if inside_icon else (255, 0, 0)
            cv2.circle(img, (x, y), 10, color, 3)
        
        images.append(img)
    
    # Add target image (without icon visualizations)
    target_img = cv2.imread(target_image)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    images.append(target_img)
    
    # Calculate grid dimensions (roughly 2:1 aspect ratio)
    total_images = history_length + 1  # including target
    cols = int(np.sqrt(total_images * 2))  # multiply by 2 for 2:1 aspect ratio
    rows = (total_images + cols - 1) // cols  # ceiling division
    
    # Create the grid
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]
    
    # Plot all images
    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        if i < len(images):
            axes[row][col].imshow(images[i])
            axes[row][col].axis('off')
            if i < history_length:
                axes[row][col].set_title(f'Frame {i+1}')
            else:
                axes[row][col].set_title('Target')
        else:
            axes[row][col].axis('off')  # Hide empty subplots
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_previous_frames(record_num, image_num, history_length):
    """Get paths of previous frames"""
    frames = []
    for i in range(history_length, 0, -1):
        prev_num = image_num - i
        if prev_num >= 0:  # Make sure we don't go below image_0
            frames.append(f"train_dataset/record_{record_num}/image_{prev_num}.png")
    return frames

def get_actions_for_sequence(mapping_dict, record_num, image_nums):
    """Get actions for a sequence of frames"""
    actions = []
    for img_num in image_nums:
        key = (record_num, img_num)
        actions.append(mapping_dict.get(key))
    return actions

def analyze_sequences(csv_path, output_dir="analysis_results", debug=False, history_length=14, time_threshold=0.6):
    """Analyze all sequences and compute accuracy"""
    # Load mapping and target frames
    print("Loading mapping dictionary...")
    with open('image_action_mapping.pkl', 'rb') as f:
        mapping_dict = pickle.load(f)
    target_df = pd.read_csv('desktop_sequences_filtered_with_desktop_1.5k_target_frames.csv')
    
    # Clean up previous results
    output_dir = Path(output_dir)
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if debug:
        print("Debug mode: using first 100 rows only")
        target_df = target_df.head(100)
    
    results = []
    error_cases = defaultdict(list)
    
    for idx, row in tqdm(target_df.iterrows(), total=len(target_df)):
        record_num = row['record_num']
        image_num = row['image_num']
        
        # Get previous frames
        image_seq = get_previous_frames(record_num, image_num, history_length)
        target_image = f"train_dataset/record_{record_num}/image_{image_num}.png"
        
        # Get corresponding actions
        prev_img_nums = [int(re.search(r'image_(\d+)', img).group(1)) if 'padding' not in img else -1 
                        for img in image_seq]
        action_seq = get_actions_for_sequence(mapping_dict, record_num, prev_img_nums + [image_num])
        
        # Get prediction and ground truth
        prediction = predict_target(action_sequence=action_seq, time_threshold=time_threshold)
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
            if len(error_cases[error_key]) < 100:  # Only store up to 100 cases per error type
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
    for true_class in CLUSTER_PATHS.keys():
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
    csv_path = "desktop_sequences_filtered_with_desktop_1.5k.csv"
    csv_path = "desktop_sequences_filtered_with_desktop_1.5k_last100.challenging.csv"
    output_dir = "desktop_analysis_results_with_desktop_1.5k_last100_challenging"
    history_length = 28  # Number of previous frames to show in transitions
    time_threshold = 0.6
    debug = True # Set to True to process only first 100 rows
    results_data = []


    rerun = False
    if rerun:
        for history_length in [2, 4, 8, 16, 32, 64, 128]:
            results_df, error_cases = analyze_sequences(
                csv_path, 
                output_dir,
                debug=debug,
                history_length=history_length,
                time_threshold=time_threshold
            )
            
            # Calculate accuracy for each icon type
            for icon in ['desktop', 'firefox', 'terminal', 'root', 'trash']:
                icon_cases = results_df[results_df['ground_truth'] == icon]
                if len(icon_cases) > 0:
                    accuracy = icon_cases['correct'].mean()
                    results_data.append({
                        'history_length': history_length,
                        'icon': icon,
                        'accuracy': accuracy,
                        'total_cases': len(icon_cases)
                    })
        # Save to CSV
        pd.DataFrame(results_data).to_csv('icon_accuracy_vs_history.csv', index=False)

    # Part 2: Create plot
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Read results
    df = pd.read_csv('icon_accuracy_vs_history.csv')

    # Create plot
    plt.figure(figsize=(10, 6))
    for icon in df['icon'].unique():
        icon_data = df[df['icon'] == icon]
        plt.plot(icon_data['history_length'], icon_data['accuracy'], 
                marker='o', label=f'{icon} (n={icon_data.iloc[0]["total_cases"]})')

    plt.xlabel('History Length')
    plt.ylabel('Accuracy')
    plt.title('Icon Detection Accuracy vs History Length')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xscale('log', base=2)  # Use log scale for history length
    plt.ylim(0, 1)  # Set y-axis from 0 to 1

    # Add horizontal lines at 0.5 and 1.0
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('icon_accuracy_vs_history.png')
    plt.close()