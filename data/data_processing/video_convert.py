import cv2
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip
from PIL import Image
import os
import argparse
from math import exp, floor

#Creates the padding image for your model as a starting point for the generation process.
def create_padding_img() -> Image.Image:
    return Image.new('RGB', (256, 256), color=(0, 0, 0))

def create_video_from_frames(frames: list, save_path: str, fps: int = 15):

    """
    Takse a sequence of generated images from the model and constructs a video.
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


def video_to_frames(video_path: str = '../raw_data/custom/videos/record_custom.mp4', save_path: str = 'train_dataset', actions_path: str = '../raw_data/custom/actions/record_custom.csv', save_map: bool = False, video_num: int = 0) -> pd.DataFrame:

    '''
    Opens a video and ouputs the frame and corresponding actions. Needs to be processed into sequences later.
    Also saves each frame in save_path folder.

    Paramters:
        video_num: The video number to label the image to. Used for conversion of multiple videos.
        save_map: If true saves the map to a csv @ save_path.
    Returns:
        df: a dataframe with image frame path and action column. Maps frames to actions.
    '''

    os.makedirs(save_path, exist_ok=True)

    mouse_data = pd.read_csv(actions_path)

    # List to hold the frames (as numpy arrays)
    images_paths = []
    actions = []

    # Open the video file
    with VideoFileClip(video_path) as video:
        #print (video.duration, video.fps)
        duration = video.duration  # Total duration in seconds
        fps = video.fps  # Frames per second
        #print (duration, int(fps * duration), len(mouse_data))
        #assert int(fps * duration) == len(mouse_data), (int(fps * duration), len(mouse_data))
        assert fps == 15, fps
        
        # Iterate through each frame
        for frame_number in range(0, int(fps * duration)):
            # Get the corresponding action directly using frame_number
            action_row = mouse_data.iloc[frame_number]

            if action_row['Right Click']:
                action = 'right_click'
            elif action_row['Left Click']:
                action = 'left_click'
            else:
                action = f"{int(action_row['X'])}~{int(action_row['Y'])}"
            
            # Get the frame at the specified time
            frame = video.get_frame(frame_number / fps)
            
            frame_rgb = frame #cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            save_dir = f'{save_path}/record_{video_num}'
            os.makedirs(save_dir, exist_ok=True)
            path = f'{save_dir}/image_{frame_number}.png'
            Image.fromarray(frame_rgb).save(path)  # Saves in the correct format

            #append the path and labels
            images_paths.append(path)
            actions.append(str(action))


    df = pd.DataFrame()
    df['Image_path'] = images_paths
    df['Action'] = actions

    #Save the mapping of frames to action if needed.
    if save_map:
        df.to_csv(os.path.join(save_dir, 'frame_action_map.csv'), index=False)

    return df

def action_binning(actions_list: list, bin_width = 4) -> list:

    """
    Converts the action sequence x,y into changes in x and y and bins them.
    Returns:
        deltas: the binned deltas.
    """

    # Split the string 'x:y' into integers and calculate deltas
    deltas = []
    prev_x, prev_y = 0, 0  # Initial point is set to (0, 0)

    #first action entry has no deltas so normalize to 0,0
    prev_x, prev_y = map(int, actions_list[0].split('~'))
    
    for action in actions_list:
        #Check if the action is a x:y pair.
        if ':' in action: 
            x, y = map(int, action.split('~'))  # Convert 'x:y' into integers
            delta_x = x - prev_x
            delta_y = y - prev_y
            deltas.append(f"{delta_x//bin_width}:{delta_y//bin_width}")
            prev_x, prev_y = x, y  # Update previous x and y
        else: deltas.append(action)

    return deltas

def sequence_creator(dataframe: pd.DataFrame, save_path: str, seq_len: int = 8, bin_width: int = 1, pad_start: bool = False, save_dataset: bool = False) -> pd.DataFrame:

    """
    Turns images and their corresponding actions into sequences of length seq_len. (seq_len - 1 cond actions and images).
    Also converts the action X,Y cordinates and bins them by bin_width. Saves the sequence dataset to save_path to be referenced by a dataset.
    
    Parameters:
        seq_len: The number of previous conditioning frames -1.
        bin_width: The divisor of the deltas in x,y.
        pad_start: Pads the start of the video with padding frames such that each frame in the video is a target. This is so we can start inference with < seq_len frames.
        save_dataset: If true saves the sequence dataset to csv @ save_path.

    Returns:
        seq_df: Dataframe with video and actions in sequences.
    """

    image_paths = dataframe['Image_path'].tolist()
    actions = dataframe['Action'].tolist() # action_binning(dataframe['Action'].tolist(), bin_width=bin_width)

    #If padding, prepend the padding image and first action to the list.
    if pad_start:
        image_paths_padding = [save_path + '/padding.png' for _ in range(seq_len - 1)]
        actions_padding = ['0~0' for _ in range(seq_len - 1)]

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
    
    parser.add_argument("--save_map", type=bool, default=False,
                        help="saves the map of frames to actions.")

    parser.add_argument("--seq_len", type=int, default=8,
                        help="This number -1 is the number of frames conditioned on.")
    
    parser.add_argument("--bin_width", type=int, default=1,
                        help="divides the x,y cordinates by this number.")

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
    sequence_creator(df, save_path, seq_len=args.seq_len, bin_width=args.bin_width)
