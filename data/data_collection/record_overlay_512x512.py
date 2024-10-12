import mss
import cv2
import numpy as np
import time
from datetime import datetime
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import pandas as pd
import numpy as np

import argparse

def overlay_mouse_actions(video_file, csv_file, output_file):
    video = VideoFileClip(video_file)
    mouse_data = pd.read_csv(csv_file)
    fps = video.fps
    video_duration = video.duration
    video_width, video_height = video.w, video.h
    
    # Function to create an image with the cursor and text overlay
    def make_frame(t):
        # Find the closest mouse action to the current time
        closest_index = mouse_data['Timestamp'].sub(t).abs().idxmin()
        closest_index = mouse_data['Timestamp'].sub(t).abs().idxmin()
        action = mouse_data.iloc[closest_index]
        
        # Read the corresponding video frame
        frame = video.get_frame(t)
        
        # Convert the frame to OpenCV format (BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Get the mouse coordinates
        x_pos = int(action['X'])
        y_pos = int(action['Y'])

        x_pos = int(x_pos - (1920 - 512)/2) #relative cordinate mapped to video
        y_pos = int(y_pos - (1080 - 512)/2) #relative cordinate mapped to video
        
        # Add text to the frame
        text = f"X: {x_pos}, Y: {y_pos}, RC: {action['Right Click']}, LC: {action['Left Click']}"
        cv2.putText(frame, text, (10, video.h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)     

        # Overlay a red dot at the (X, Y) position
        if 0 <= x_pos <= video.w and 0 <= y_pos <= video.h:
            cv2.circle(frame, (x_pos, y_pos), 3, (0, 0, 255), -1)  # Red dot with radius 10
        
        return frame

    # Create the video writer for the output video
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (video.w, video.h))
    
    # Process each frame and write it to the output video
    for t in np.arange(0, video_duration, 1 / fps):
        frame = make_frame(t)
        out.write(frame)
    
    out.release()


def record_screen(output_file, fps=12, duration=6):
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Select the primary monitor
        interval = 1 / fps
        end_time = time.time() + duration
        
        # Get the screen dimensions
        screen_width = monitor['width']
        screen_height = monitor['height']
        
        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .avi files
        out = cv2.VideoWriter(output_file, fourcc, fps, (screen_width, screen_height))
        
        while time.time() < end_time:
            start_time = time.time()
            img = np.array(sct.grab(monitor))  # Capture the screen
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert from BGRA to BGR
            
            # Write the frame to the video file
            out.write(frame)
            
            # Optionally display the frame
            # cv2.imshow("Screen", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
            # Sleep to maintain the desired fps
            time.sleep(max(0, interval - (time.time() - start_time)))
        
        # Release the VideoWriter object and close the OpenCV window
        out.release()
        cv2.destroyAllWindows()

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
    # record_screen(output_file=output_file)
    overlay_mouse_actions(output_file=output_file, video_file=args.video_file, csv_file=args.mouse_input_csv)