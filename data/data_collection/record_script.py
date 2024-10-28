import mss
import cv2
import numpy as np
import os
import asyncio
#from dotenv import load_dotenv
import time
from pynput.mouse import Controller, Listener, Button
import pandas as pd
import pyautogui

# Load environment variables
#load_dotenv()
current_directory = os.path.dirname(os.path.abspath(__file__)) #for obs video paths
parent_directory = os.path.dirname(current_directory)

mouse = Controller()
right_click = False
left_click = False


def on_click(x, y, button, pressed):
    global right_click, left_click
    if button == Button.right: right_click = pressed
    elif button == Button.left: left_click = pressed

def draw_cursor(frame, x, y, left_click=False, right_click=False, scaling_factor=1):
    """Draw a cursor on the frame at the given position"""
    # Convert coordinates to integers and apply scaling
    x, y = int(x * scaling_factor), int(y * scaling_factor)
    
    # Draw cursor (white with black border for visibility)
    cursor_size = 50
    # Outer black circle
    cv2.circle(frame, (x, y), cursor_size + 1, (0, 0, 0), 2)
    # Inner white circle, filled if clicked
    if left_click or right_click:
        cv2.circle(frame, (x, y), cursor_size, (255, 255, 255), -1)
    else:
        cv2.circle(frame, (x, y), cursor_size, (255, 255, 255), 2)
    return frame

async def record(save_dir: str = 'raw_data', save_name: str = 'record_0', 
                duration: int = 12, function_to_record: callable = None, 
                fn_args: tuple = ()):
    """
    Records mouse positions, clicks, and screen at 15 fps.
    Frames are resized to 256x256 for storage efficiency.
    """
    fps = 15
    interval = 1.0 / fps
    output_size = (256, 256)  # Target size for video frames
    
    # Ensure directories exist with parent_directory
    os.makedirs(f'{parent_directory}/{save_dir}/videos', exist_ok=True)
    os.makedirs(f'{parent_directory}/{save_dir}/actions', exist_ok=True)

    # Initialize mouse data collection
    data = []
    
    # Start mouse listener
    listener = Listener(on_click=on_click)
    listener.start()

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        scaling_factor = 1
        
        # Get actual frame dimensions first
        screen = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
        height, width = frame.shape[:2]
        print(f"Frame shape: {frame.shape}")
        
        # Calculate scaling factor by comparing monitor and frame dimensions
        scaling_factor = height / monitor['height']
        print(f"Detected scaling factor: {scaling_factor}")
        
        # Initialize video writer with output_size
        codec = 'mp4v'
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(
            f'{parent_directory}/{save_dir}/videos/{save_name}.mp4', 
            fourcc, 
            fps, 
            output_size,  # Use smaller size
            isColor=True
        )
           
        if not out.isOpened():
            raise Exception("Could not open video writer")
        
        print(f"Monitor settings: {monitor}")
        print(f"Original dimensions: width={width}, height={height}")
        print(f"Output dimensions: {output_size}")
        
        frame_count = 0
        try:
            # Start function_to_record if provided
            if function_to_record:
                function_task = asyncio.create_task(asyncio.to_thread(function_to_record, *fn_args))
                # Wait for the function to start
                await asyncio.sleep(0.1)  # Small delay to ensure function starts

            start_time = time.time()
            while time.time() - start_time < duration:
                frame_start = time.time()

                # Capture screen and mouse data
                screen = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
                
                # Draw cursor before resizing
                x, y = mouse.position
                frame = draw_cursor(frame, x, y, left_click, right_click, scaling_factor)
                
                # Resize frame
                frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
                
                # Write resized frame
                success = out.write(frame)
                frame_count += 1
                #if frame_count % 15 == 0:  # Print every second
                #    print(f"Wrote frame {frame_count}, success: {success}")

                current_time = time.time() - start_time
                seconds = int(current_time)
                milliseconds = int((current_time - seconds) * 1000)
                time_formatted = f"{seconds}:{milliseconds}"

               #x, y = mouse.position
                data.append([
                    current_time, 
                    time_formatted, 
                    x, y, 
                    right_click, 
                    left_click
                ])

                # Maintain 15 fps
                elapsed = time.time() - frame_start
                if elapsed < interval:
                    await asyncio.sleep(interval - elapsed)

        finally:
            # Ensure proper cleanup
            if out.isOpened():
                out.release()
                print("Video writer released")
            listener.stop()
            cv2.destroyAllWindows()  # Clean up any OpenCV windows

            # Wait for function_to_record to complete if it exists
            if function_to_record and 'function_task' in locals():
                try:
                    await function_task
                except asyncio.CancelledError:
                    pass

            # Cancel function_to_record if it's still running
            if function_to_record and 'function_task' in locals():
                if not function_task.done():
                    function_task.cancel()
                    try:
                        await function_task
                    except asyncio.CancelledError:
                        pass

            # Save mouse data with parent_directory
            df = pd.DataFrame(
                data, 
                columns=['Timestamp', 'Timestamp_formated', 'X', 'Y', 'Right Click', 'Left Click']
            )
            df.to_csv(
                f'{parent_directory}/{save_dir}/actions/{save_name}.csv', 
                index=False
            )

        #max_x, max_y = pyautogui.size()
        #print(f"Recorded mouse data on: width {max_x}px height {max_y}px screen.")
        print(f"Saved video and actions csv at {save_dir}/{save_name}.")

        # Verify the video file was created successfully
        if os.path.exists(f'{parent_directory}/{save_dir}/videos/{save_name}.mp4'):
            print(f"Video file size: {os.path.getsize(f'{parent_directory}/{save_dir}/videos/{save_name}.mp4')} bytes")
        else:
            print("Video file was not created!")
