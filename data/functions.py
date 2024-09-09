import mss
import cv2
import numpy as np
import time
from datetime import datetime
import pyautogui

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

        # Load cursor image
        cursor = cv2.imread('cursor.png', cv2.IMREAD_UNCHANGED)
        cursor_height, cursor_width = cursor.shape[:2]
        
        while time.time() < end_time:
            start_time = time.time()
            img = np.array(sct.grab(monitor))  # Capture the screen
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert from BGRA to BGR

            # Get mouse position
            mouse_x, mouse_y = pyautogui.position()
            
            # Ensure cursor is within screen bounds
            if 0 <= mouse_x < screen_width and 0 <= mouse_y < screen_height:
                # Overlay cursor on the frame
                cursor_x = mouse_x - cursor_width // 2
                cursor_y = mouse_y - cursor_height // 2
                
                # Create a mask for cursor
                cursor_mask = cursor[:, :, 3] / 255.0
                cursor_rgb = cursor[:, :, :3]
                
                # Place cursor on frame
                for c in range(3):  # Loop over color channels
                    frame[cursor_y:cursor_y + cursor_height, cursor_x:cursor_x + cursor_width, c] = (
                        cursor_mask * cursor_rgb[:, :, c] +
                        (1 - cursor_mask) * frame[cursor_y:cursor_y + cursor_height, cursor_x:cursor_x + cursor_width, c]
                    )
            
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
    record_screen(output_file=output_file)