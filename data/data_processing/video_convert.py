import cv2
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip
from PIL import Image
import os
from math import exp, floor

def create_video_from_frames(frames: list, save_path: str):

    """
    Takse a sequence of generated images from the model and constructs a video.
    """

    image = frames[0]

    W, H = image.size

    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (W, H))
    for frame in frames:
        frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()

    print(f"\u2705 Saved video at {save_path}")


def video_to_frames(video_path: str, save_path: str, actions_path: str):

    '''
    Opens a video and ouputs the frame and corresponding actions. Needs to be processed into sequences later.
    Also saves each frame in save_path folder.
    Returns:
        df: a dataframe with image frame path and action column.
    '''

    os.makedirs(save_path, exist_ok=True)

    mouse_data = pd.read_csv(actions_path)

    # List to hold the frames (as numpy arrays)
    images_paths = []
    actions = []

    # Open the video file
    with VideoFileClip(video_path) as video:
        duration = int(video.duration)  # Total duration in seconds
        fps = video.fps  # Frames per second
        
        # Iterate through each frame
        for frame_number in range(int(fps * duration)):
            # Calculate the time in seconds for the current frame
            time = frame_number / fps

            #Map actions to frames
            closest_idx = mouse_data['Timestamp'].sub(time).abs().idxmin()
            closest_action = mouse_data.iloc[closest_idx]

            if closest_action['Right Click']:
                action = 'right_click'
            elif closest_action['Left Click']:
                action = 'left_click'
            else: #Its a move.
                action = f"{closest_action['X']}:{closest_action['Y']}"
            
            # Get the frame at the specified time
            frame = video.get_frame(time)
            
            frame_rgb = frame #cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            path = f'{save_path}/image_{frame_number}.png'
            Image.fromarray(frame_rgb).save(path)  # Saves in the correct format

            #append the path and labels
            images_paths.append(path)
            actions.append(str(action))


    df = pd.DataFrame()
    df['Image_path'] = images_paths
    df['Action'] = actions
    df.to_csv(os.path.join(save_path, 'train_info.csv'), index=False)

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
    prev_x, prev_y = map(int, actions_list[0].split(':'))
    
    for action in actions_list:
        #Check if the action is a x:y pair.
        if ':' in action: 
            x, y = map(int, action.split(':'))  # Convert 'x:y' into integers
            delta_x = x - prev_x
            delta_y = y - prev_y
            deltas.append(f"{delta_x//bin_width}:{delta_y//bin_width}")
            prev_x, prev_y = x, y  # Update previous x and y
        else: deltas.append(action)

    return deltas

def sequence_creator(dataframe: pd.DataFrame, save_path: str, seq_len: int = 8, bin_width: int = 1):

    """
    Turns images and their corresponding actions into sequences of length seq_len. (seq_len - 1 cond actions and images).
    Also converts the action X,Y cordinates and bins them by bin_width. Saves the sequence dataset to save_path to be referenced by a dataset.
    """

    image_paths = dataframe['Image_path'].tolist()
    actions = action_binning(dataframe['Action'].tolist(), bin_width=bin_width)

    seq_df = pd.DataFrame() #holds the new sequences
    context_cond_images = []
    target_image = []
    context_actions = []

    for i in range(len(df) - seq_len + 1):
        context_cond_images.append(image_paths[i:i+seq_len-1])
        context_actions.append(actions[i:i+seq_len])
        target_image.append(image_paths[i+seq_len - 1])

    seq_df['Image_seq_cond_path'] = context_cond_images
    seq_df['Action_seq'] = context_actions
    seq_df['Target_image'] = target_image
    seq_df.to_csv(os.path.join(save_path, 'train_sequence_info.csv'), index=False)




if __name__ == "__main__":

    save_path='train_256x256_w_actions_binned'
    
    df = video_to_frames(
        video_path='sample11_256x256.mp4',
        save_path=save_path,
        actions_path='mouse_actions11.csv'
    )

    #Saves the sequence dataset file
    sequence_creator(df, save_path, seq_len=8, bin_width=4)