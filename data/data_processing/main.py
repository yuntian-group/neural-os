import pandas as pd
import os, argparse
from PIL import Image
from tqdm import tqdm
import multiprocessing
from functools import partial
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import ast
import pickle


#Creates the padding image for your model as a starting point for the generation process.
def create_padding_img(width, height):
    """Creates a black image for padding with specified dimensions"""
    return Image.new('RGB', (width, height), color='black')


def compute_distance(current_frame, prev_frame):
    current_frame = np.array(current_frame)
    prev_frame = np.array(prev_frame)

    current_norm = current_frame.astype(float) / 255.0
    prev_norm = prev_frame.astype(float) / 255.0

    mse = np.mean((current_norm - prev_norm) ** 2)
    return mse


def parse_args():
    parser = argparse.ArgumentParser(description="Converts a group of videos and their respective actions into one training dataset.")
    
    parser.add_argument("--save_dir", type=str, default='../data_collection/raw_data/train_dataset',
                        help="directory to save the entire training set.")

    parser.add_argument("--video_dir", type=str, default='../data_collection/raw_data/raw_data/videos',
                        help="directory where the videos are saved.")
    
    parser.add_argument("--actions_dir", type=str, default='../data_collection/raw_data/raw_data/actions',
                        help="directory where the actions are saved.")
                            
    parser.add_argument("--filter_videos", action='store_true',
                        help="Whether to filter videos based on their dimensions.")
    parser.add_argument("--seq_len", type=int, default=32,
                        help="The length of the sequence to process.")
    parser.set_defaults(filter_videos=False)
    args = parser.parse_args()
    print (args)
    if args.filter_videos:
        print ("Filtering videos based on their dimensions.")
        args.save_dir = args.save_dir + "_filtered"
    # Add directory validation
    for dir_path in [args.video_dir, args.actions_dir]:
        if not os.path.exists(dir_path):
            raise ValueError(f"Directory does not exist: {dir_path}")
            
    return args

def process_video(record_num: int, args: argparse.Namespace, save_dir: str, video_files: list[str]) -> pd.DataFrame | None:
    #import pdb; pdb.set_trace()
    video_file = f'record_{record_num}.mp4'
    if video_file not in video_files:
        return None
    video_path = os.path.join(args.video_dir, video_file)
    actions_path = os.path.join(args.actions_dir, f'record_{record_num}.csv')
    filter_videos = args.filter_videos
    seq_len = args.seq_len
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(actions_path):
        raise FileNotFoundError(f"Actions file not found: {actions_path}")

    mouse_data = pd.read_csv(actions_path)
    # List to hold the frames (as numpy arrays)
    mapping_dict = {}
    target_data = []
    # Open the video file
    with VideoFileClip(video_path) as video:
        fps = video.fps
        duration = video.duration
        # Make FPS check a warning instead of hard assertion
        if fps != 15:
            print(f"Warning: Expected FPS of 15, got {fps}")
            assert False

        # First pass: identify frames to keep
        frames_to_keep = []
        all_frames = []
        prev_frame = None
        down_keys = set([])
        
        for image_num in range(0, int(fps * duration)):
            action_row = mouse_data.iloc[image_num]
            x = int(action_row['X'])
            y = int(action_row['Y'])
            left_click = True if action_row['Left Click'] == 1 else False
            right_click = True if action_row['Right Click'] == 1 else False
            key_events = ast.literal_eval(action_row['Key Events'])
            
            current_frame = Image.fromarray(video.get_frame(image_num / fps))
            all_frames.append(current_frame)
            
            # Track key states
            for key_state, key in key_events:
                if key_state == 'keydown':
                    down_keys.add(key)
                elif key_state == 'keyup':
                    down_keys.remove(key)
                else:
                    raise ValueError(f"Unknown key state: {key_state}")
            
            # Store mapping regardless of filtering
            mapping_dict[(record_num, image_num)] = (x, y, left_click, right_click, list(down_keys))
            
            filtered = False
            if filter_videos:
                if prev_frame is None:
                    filtered = True
                else:
                    distance = compute_distance(current_frame, prev_frame)
                    if distance < 0.1:
                        filtered = True
            
            prev_frame = current_frame
            
            if not filtered:
                frames_to_keep.append(image_num)
        
        # Second pass: save frames and their sequences
        for keep_frame in frames_to_keep:
            # Save the current frame that we want to keep
            record_dir = f'{save_dir}/record_{record_num}'
            os.makedirs(record_dir, exist_ok=True)
            
            # Save this frame
            all_frames[keep_frame].save(f'{record_dir}/image_{keep_frame}.png')
            
            # Save the past seq_len frames
            start_idx = max(0, keep_frame - seq_len)
            for seq_idx in range(start_idx, keep_frame):
                save_path = f'{record_dir}/image_{seq_idx}.png'
                if not os.path.exists(save_path):
                    all_frames[seq_idx].save(save_path)
            
            # Add the current frame to target data
            target_data.append((record_num, keep_frame))

    return (target_data, mapping_dict)

def get_video_dimensions(video_path: str) -> tuple[int, int]:
    """
    Safely get video dimensions using OpenCV.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple of (width, height)
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        ret, frame = cap.read()
        cap.release()  # Properly release the video capture
        
        if not ret:
            raise ValueError(f"Could not read frame from video file: {video_path}")
        
        height, width = frame.shape[:2]
        return width, height
        
    except Exception as e:
        raise ValueError(f"Error reading video dimensions: {str(e)}")

if __name__ == "__main__":
    """
    Processes multiple videos sequentially and creates an entire processed dataset.
    """

    args = parse_args()
    
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Improve video file pattern matching
    video_files = [f for f in os.listdir(args.video_dir) 
                  if f.startswith('record_') and f.endswith('.mp4')
                  and os.path.isfile(os.path.join(args.video_dir, f))]

    #video_files = video_files[:10]
    
    if not video_files:
        raise ValueError(f"No valid video files found in {args.video_dir}")
    
    # Get video dimensions using the new helper function
    first_video = os.path.join(args.video_dir, video_files[0])
    width, height = get_video_dimensions(first_video)
    print(f"Video dimensions: width: {width}, height: {height}")

    # Create a padding image with the same size
    padding_image = create_padding_img(width, height)
    padding_image.save(os.path.join(save_dir, 'padding.png'))

    all_seqs = []

    n = len(video_files)
    print(f"Processing {n} videos.")

    # Create a partial function with fixed arguments
    process_video_partial = partial(process_video, args=args, save_dir=save_dir, video_files=video_files)

    # Use all available CPU cores (removed hardcoding to 1)
    num_workers = min(multiprocessing.cpu_count(), 150)
    print(f"Using {num_workers} workers.")

    try:
        # Create a multiprocessing pool
        with multiprocessing.Pool(num_workers) as pool:
            # Process videos in parallel
            results = list(tqdm(
                pool.imap(process_video_partial, range(n)), 
                total=n, 
                desc="Processing videos", 
                unit="video"
            ))

        # Filter out None results and combine sequences
        all_seqs = [item for result in results if result is not None for item in result[0]]

        all_mapping_dict = {}
        for result in results:
            mapping_dict = result[1]
            for key, value in mapping_dict.items():
                all_mapping_dict[key] = value

        # save to a csv of two columns, record_num and image_num
        all_seqs_df = pd.DataFrame(all_seqs, columns=['record_num', 'image_num'])
        all_seqs_df.to_csv(os.path.join(save_dir, 'train_dataset.target_frames.csv'), index=False)
        with open(os.path.join(save_dir, 'image_action_mapping_with_key_states.pkl'), 'wb') as f:
            pickle.dump(all_mapping_dict, f)
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")