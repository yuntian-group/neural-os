import mss
import cv2
import numpy as np
import time
from datetime import datetime
import pyautogui
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import pandas as pd
import numpy as np

def overlay_mouse_actions(video_file, csv_file, output_file):
    video = VideoFileClip(video_file)
    mouse_data = pd.read_csv(csv_file)
    fps = video.fps
    
    def make_frame(t):
        # Find the closest mouse action to the current time
        closest_index = mouse_data['Timestamp'].sub(t).abs().idxmin()
        action = mouse_data.iloc[closest_index]
        
        # Create a text clip for the mouse position
        txt_clip = TextClip(f"X: {action['X']}, Y: {action['Y']}, Clicked: {action['Clicked']}", fontsize=24, color='white')
        txt_clip = txt_clip.set_position(('center', 'bottom')).set_duration(video.duration)
        
        # Create a background clip to overlay text on
        background = np.zeros((video.h, video.w, 3), dtype='uint8')
        background[:] = 0  # Black background
        return txt_clip.set_duration(video.duration).set_fps(fps).get_frame(t)
    
    txt_clips = [make_frame(t) for t in np.arange(0, video.duration, 1 / fps)]
    final_clip = CompositeVideoClip([video] + txt_clips)
    final_clip.write_videofile(output_file, codec='libx264')


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


if __name__ == "__main__":
    output_file=f'record_{datetime.now().strftime("%Y-%m-%d")}.mp4'
    # record_screen(output_file=output_file)
    overlay_mouse_actions('your_video.mp4', 'mouse_actions.csv', 'output_with_mouse.mp4')