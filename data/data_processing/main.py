import pandas as pd
import os, argparse
from PIL import Image
from tqdm import tqdm
import multiprocessing
from functools import partial
import cv2

import cv2
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip
from PIL import Image
import os
import argparse
from math import exp, floor
import ast


def extract_numbers(path):
    """Extract record and image numbers from path"""
    match = re.search(r'record_(\d+)/image_(\d+)', path)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

def format_action(x: int, y: int, left_click: bool, right_click: bool, key_events: str, is_padding: bool = False) -> str:
    """
    Format mouse action data into a standardized string format.
    
    Args:
        x: X coordinate
        y: Y coordinate
        left_click: Left mouse button state
        right_click: Right mouse button state
        key_events: Key events
        is_padding: Whether this is a padding action
        
    Returns:
        Formatted action string
    """
    key_events = ast.literal_eval(key_events)
    if is_padding:
        x, y, left_click, right_click, key_events = 0, 0, False, False, []
    formatted_action = (int(x), int(y), True if left_click else False, True if right_click else False, key_events)
    return formatted_action

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

def video_to_frames(video_path: str, actions_path: str, record_num: int, save_dir: str = 'train_dataset', filter_videos: bool = False) -> pd.DataFrame:
    # Add input validation
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

        prev_frame = None
        down_keys = set([])
        for image_num in range(0, int(fps * duration)):
            action_row = mouse_data.iloc[image_num]
            x = action_row['X']
            y = action_row['Y']
            left_click = action_row['Left Click']
            right_click = action_row['Right Click']
            key_events = action_row['Key Events']
            
            current_frame = Image.fromarray(video.get_frame(image_num / fps))
            if filter_videos:
                if prev_frame is None:
                    filtered = True
                if compute_distance(current_frame, prev_frame) < 0.1:
                    filtered = True
            for key_state, key in key_events:
                if key_state == 'keydown':
                    down_keys.add(key)
                elif key_state == 'keyup':
                    down_keys.remove(key)
                else:
                    raise ValueError(f"Unknown key state: {key_state}")
            prev_frame = current_frame
            if filtered:
                continue
            record_dir = f'{save_dir}/record_{record_num}'
            path = f'{record_dir}/image_{image_num}.png'
            os.makedirs(record_dir, exist_ok=True)
            current_frame.save(path)
            mapping_dict[(record_num, image_num)] =  (x, y, left_click, right_click, list(down_keys))
            #actions.append(action)
    #If padding, prepend the padding image and first action to the list.
    
    #image_paths_padding = [save_path + '/padding.png' for _ in range(seq_len - 1)]
    #actions_padding = [format_action(0, 0, False, False, '[]', is_padding=True) for _ in range(seq_len - 1)]

    #prepend the padding if needed.
    #image_paths = image_paths_padding + image_paths
    #actions = actions_padding + actions

    #target_data = []
    #for i in range(len(image_paths)):
    #    record_num, image_num = extract_numbers(image_paths[i])
    #    target_data.append((record_num, image_num))


    return (target_data, mapping_dict)

def parse_args():
    parser = argparse.ArgumentParser(description="Converts a group of videos and their respective actions into one training dataset.")
    
    parser.add_argument("--save_dir", type=str, default='train_dataset',
                        help="directory to save the entire training set.")

    parser.add_argument("--video_dir", type=str, default='../data_collection/raw_data/raw_data/videos',
                        help="directory where the videos are saved.")
    
    parser.add_argument("--actions_dir", type=str, default='../data_collection/raw_data/raw_data/actions',
                        help="directory where the actions are saved.")
    
    parser.add_argument("--seq_len", type=int, default=2,
                        help="This number -1 is the number of frames conditioned on.")
                        
    parser.add_argument("--filter_videos", action='store_true',
                        help="Whether to filter videos based on their dimensions.")
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

def process_video(i: int, args: argparse.Namespace, save_dir: str, video_files: list[str]) -> pd.DataFrame | None:
    """
    Process a single video file and create sequences.
    
    Args:
        i: Video index
        args: Command line arguments
        save_dir: Directory to save processed data
        video_files: List of available video files
        
    Returns:
        DataFrame containing sequences or None if video not found
    """
    import pdb; pdb.set_trace()
    video_file = f'record_{i}.mp4'
    if video_file not in video_files:
        return None

    try:
        df = video_to_frames(
            video_path=os.path.join(args.video_dir, video_file),
            save_path=save_dir,
            actions_path=os.path.join(args.actions_dir, f'record_{i}.csv'),
            video_num=i,
            filter_videos=args.filter_videos,
            seq_len=args.seq_len
        )
        
        #seq_df = sequence_creator(df, save_dir, seq_len=args.seq_len)
        #del df  # Clean up memory
        return seq_df
        
    except Exception as e:
        print(f"Error processing video {video_file}: {str(e)}")
        raise e

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

    video_files = video_files[:10]
    
    if not video_files:
        raise ValueError(f"No valid video files found in {args.video_dir}")
    
    # Get video dimensions using the new helper function
    first_video = os.path.join(args.video_dir, video_files[0])
    width, height = get_video_dimensions(first_video)
    print(f"Video dimensions: width: {width}, height: {height}")

    # Create a padding image with the same size
    black_image = create_padding_img(width, height)
    black_image.save(os.path.join(save_dir, 'padding.png'))

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
        debug = True
        if debug:
            results = [process_video_partial(0), process_video_partial(1), process_video_partial(2)]
        else:
            with multiprocessing.Pool(num_workers) as pool:
                # Process videos in parallel
            results = list(tqdm(
                pool.imap(process_video_partial, range(n)), 
                total=n, 
                desc="Processing videos", 
                unit="video"
            ))

        # Filter out None results and combine sequences
        all_seqs = [item for item in result[0] for result in results if target_data is not None]
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
        #if not all_seqs:
        #    raise ValueError("No sequences were successfully processed")
            
        #all_seqs_df = pd.concat(all_seqs, ignore_index=True)
        #all_seqs_df.to_csv(os.path.join(save_dir, 'train_dataset.csv'))
        
        # Clean up memory
        #del results
        #del all_seqs
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
