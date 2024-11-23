import cv2
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip
from PIL import Image
import os
import argparse
from math import exp, floor



def format_action(x: int, y: int, left_click: bool, right_click: bool, is_padding: bool = False) -> str:
    """
    Format mouse action data into a standardized string format.
    
    Args:
        x: X coordinate
        y: Y coordinate
        left_click: Left mouse button state
        right_click: Right mouse button state
        is_padding: Whether this is a padding action
        
    Returns:
        Formatted action string
    """
    if is_padding:
        return "N N N N N N : N N N N N"
    if left_click and not right_click:
        prefix = 'L'
    elif right_click and not left_click:
        prefix = 'R'
    elif right_click and left_click:
        prefix = 'B'
    else:
        prefix = 'N'
    
    # Convert numbers to padded strings and add spaces between digits
    x_str = f"{abs(x):04d}"
    y_str = f"{abs(y):04d}"
    x_spaced = ' '.join(x_str)
    y_spaced = ' '.join(y_str)
    
    # Format with sign and proper spacing
    return prefix + ' ' + f"{'+ ' if x >= 0 else '- '}{x_spaced} : {'+ ' if y >= 0 else '- '}{y_spaced}"

#Creates the padding image for your model as a starting point for the generation process.
def create_padding_img(width, height):
    """Creates a black image for padding with specified dimensions"""
    return Image.new('RGB', (width, height), color='black')

def create_video_from_frames(frames: list[Image.Image], save_path: str, fps: int = 15) -> None:
    """
    Takes a sequence of generated images from the model and constructs a video.
    
    Args:
        frames: List of PIL Image objects
        save_path: Path where the video will be saved
        fps: Frames per second for the output video
    """

    image = frames[0]

    W, H = image.size

    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for frame in frames:
        frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()

    print(f"\u2705 Saved video at {save_path}")


def video_to_frames(video_path: str, actions_path: str, video_num: int, save_path: str = 'train_dataset') -> pd.DataFrame:
    """
    Opens a video and ouputs the frame and corresponding actions. Needs to be processed into sequences later.
    Also saves each frame in save_path folder.

    Paramters:
        video_path: str
            Path to the video file to be processed
        actions_path: str
            Path to the CSV file containing mouse action data
        save_path: str
            Directory path where extracted frames will be saved
        video_num: int
            Video number identifier used in frame filenames
    Returns:
        df: a dataframe with image frame path and action column. Maps frames to actions.
    """

    # Add input validation
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(actions_path):
        raise FileNotFoundError(f"Actions file not found: {actions_path}")

    os.makedirs(save_path, exist_ok=True)

    mouse_data = pd.read_csv(actions_path)

    # List to hold the frames (as numpy arrays)
    images_paths = []
    actions = []

    # Open the video file
    with VideoFileClip(video_path) as video:
        fps = video.fps
        duration = video.duration
        
        # Make FPS check a warning instead of hard assertion
        if fps != 24:
            print(f"Warning: Expected FPS of 24, got {fps}")
        
        for frame_number in range(0, int(fps * duration)):
            action_row = mouse_data.iloc[frame_number]
            action = format_action(action_row['X'], action_row['Y'], 
                                 action_row['Left Click'], 
                                 action_row['Right Click'], 
                                 is_padding=False)
            
            save_dir = f'{save_path}/record_{video_num}'
            path = f'{save_dir}/image_{frame_number}.png'
            
            # Either always save or make it a parameter
            os.makedirs(save_dir, exist_ok=True)
            frame = video.get_frame(frame_number / fps)
            Image.fromarray(frame).save(path)

            images_paths.append(path)
            actions.append(str(action))


    df = pd.DataFrame()
    df['Image_path'] = images_paths
    df['Action'] = actions
    return df


def sequence_creator(dataframe: pd.DataFrame, save_path: str, seq_len: int, save_dataset: bool) -> pd.DataFrame:
    """
    Creates sequences from video frames and actions.
    
    Args:
        dataframe: Input DataFrame with Image_path and Action columns
        save_path: Directory to save the dataset
        seq_len: Length of sequences to create
        save_dataset: Whether to save the sequences to disk
        
    Returns:
        DataFrame containing sequences
    """
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")

    image_paths = dataframe['Image_path'].tolist()
    actions = dataframe['Action'].tolist()

    #If padding, prepend the padding image and first action to the list.
    
    image_paths_padding = [save_path + '/padding.png' for _ in range(seq_len - 1)]
    actions_padding = [format_action(0, 0, False, False, is_padding=True) for _ in range(seq_len - 1)]

    #prepend the padding if needed.
    image_paths = image_paths_padding + image_paths
    actions = actions_padding + actions

    seq_df = pd.DataFrame() #holds the new sequences
    context_cond_images = []
    target_image = []
    context_actions = []

    for i in range(len(image_paths) - seq_len + 1):
        context_cond_images.append(image_paths[i:i+seq_len-1])
        context_actions.append(actions[i:i+seq_len])
        target_image.append(image_paths[i+seq_len-1])

    seq_df['Image_seq_cond_path'] = context_cond_images
    seq_df['Action_seq'] = context_actions
    seq_df['Target_image'] = target_image

    #Saves an individual video if specified
    if save_dataset: 
        seq_df.to_csv(os.path.join(save_path, 'train_dataset.csv'), index=False)

    return seq_df


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a video mp4 and actions csv into a training ready dataset.")
    
    parser.add_argument("--video_path", type=str, default='../raw_data/custom/videos/record_custom.mp4',
                        help="path of the source video.")

    parser.add_argument("--save_path", type=str, default='train_dataset',
                        help="where to save the dataset.")
    
    parser.add_argument("--actions_path", type=str, default='../raw_data/custom/actions/record_custom.csv',
                        help="where to save the dataset.")

    parser.add_argument("--seq_len", type=int, default=49,
                        help="This number -1 is the number of frames conditioned on.")
    
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    save_path=args.save_path

    df = video_to_frames(
        video_path=args.video_path,
        save_path=save_path,
        actions_path=args.actions_path
    )
    
    #Saves the sequence dataset file
    sequence_creator(df, save_path, seq_len=args.seq_len)
