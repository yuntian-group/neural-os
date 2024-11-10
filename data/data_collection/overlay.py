import mss
import cv2
import numpy as np
import time
from datetime import datetime
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import pandas as pd
import numpy as np
import pyautogui

import argparse

def overlay_mouse_actions(video_file, csv_file, output_file):
    video = VideoFileClip(video_file)
    mouse_data = pd.read_csv(csv_file)
    fps = video.fps
    total_frames = int(video.duration * fps)
    video_width, video_height = video.w, video.h
    
    # Function to create an image with the cursor and text overlay
    def make_frame(frame_number):
        # Get the corresponding mouse action directly
        if frame_number < len(mouse_data):
            action = mouse_data.iloc[frame_number]
        else:
            assert False, "Frame number out of range"
            action = mouse_data.iloc[-1]
        
        # Read the corresponding video frame
        t = frame_number / fps  # Convert frame number to time for video.get_frame
        frame = video.get_frame(t)
        
        # Convert the frame to OpenCV format (BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Add text to the frame
        text = f"(MOUSE INPUTS) X: {action['X']}, Y: {action['Y']}, Right Click: {action['Right Click']}, Left Click: {action['Left Click']}"
        cv2.putText(frame, text, (50, video.h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Get the mouse coordinates
        x_pos = int(action['X'])
        y_pos = int(action['Y'])
        screen_width, screen_height = pyautogui.size()
        screen_width, screen_height = 1920, 1080
        #x_pos = int(video_width / screen_width * x_pos)
        #y_pos = int(video_height / screen_height * y_pos)
        #print (screen_width, screen_height, video_width, video_height)
        x_pos = x_pos - (screen_width / 2 - video_width / 2)
        y_pos = y_pos - (screen_height / 2 - video_height / 2)
        x_pos = int(x_pos)
        y_pos = int(y_pos)
        # Draw base red dot for cursor position
        cv2.circle(frame, (x_pos, y_pos), 5, (0, 0, 255), -1)  # Red dot with radius 5
        
        # Add green circle effect for left clicks
        if action['Left Click']:
            # Draw outer circle with animation based on frame number
            radius = 10 + int(5 * np.sin(frame_number * 0.2))  # Pulsing effect
            cv2.circle(frame, (x_pos, y_pos), radius, (0, 255, 0), 2)  # Green circle outline
            # Optional: Add inner filled circle
            cv2.circle(frame, (x_pos, y_pos), 7, (0, 255, 0), -1)  # Smaller green dot
        
        return frame

    # Create the video writer for the output video
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (video.w, video.h))
    
    # Process each frame and write it to the output video
    for frame_number in range(total_frames):
        frame = make_frame(frame_number)
        out.write(frame)
    
    out.release()

def parse_args():
    parser = argparse.ArgumentParser(description="Overlay video with mouse input data. Used to verify sync between mouse input and video.")

    parser.add_argument('--video_file', type=str, required=True)
    parser.add_argument('--mouse_input_csv', type=str, required=True)
    parser.add_argument('--name', type=str, default="")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    name = args.name + "_" if args.name else ""

    output_file=f'video_with_overlay_{name}{datetime.now().strftime("%Y-%m-%d")}.mp4'
    overlay_mouse_actions(output_file=output_file, video_file=args.video_file, csv_file=args.mouse_input_csv)
