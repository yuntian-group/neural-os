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
import pkg_resources
from PIL import Image, ImageDraw
from cairosvg import svg2png
from io import BytesIO

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

def get_cursor_image():
    """Get cursor image from SVG"""
    cursor_svg = '''<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px"
     viewBox="8 4 12 20" enable-background="new 8 4 12 20" xml:space="preserve">
<polygon fill="#FFFFFF" points="8.2,20.9 8.2,4.9 19.8,16.5 13,16.5 12.6,16.6 "/>
<polygon fill="#FFFFFF" points="17.3,21.6 13.7,23.1 9,12 12.7,10.5 "/>
<rect x="12.5" y="13.6" transform="matrix(0.9221 -0.3871 0.3871 0.9221 -5.7605 6.5909)" width="2" height="8"/>
<polygon points="9.2,7.3 9.2,18.5 12.2,15.6 12.6,15.5 17.4,15.5 "/>
</svg>'''
    
    # Convert SVG to PNG in memory with larger size
    #png_data = svg2png(bytestring=cursor_svg.encode('utf-8'), 
    #                  output_width=48,  # Adjust size as needed
    #                  output_height=80)  # Maintain aspect ratio
    png_data = svg2png(bytestring=cursor_svg.encode('utf-8'), 
                      output_width=96,  # Adjust size as needed
                      output_height=160)  # Maintain aspect ratio
    
    # Convert PNG to numpy array
    cursor = Image.open(BytesIO(png_data))
    cursor_array = np.array(cursor)
    
    # Add black outline
    cursor_with_outline = Image.new('RGBA', cursor.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(cursor_with_outline)
    
    # Convert the white cursor to black outline
    black_mask = np.all(cursor_array == [255, 255, 255, 255], axis=-1)
    outline_positions = np.where(black_mask)
    
    # Draw black outline around white areas
    for y, x in zip(*outline_positions):
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < cursor.width and 0 <= new_y < cursor.height:
                if not black_mask[new_y, new_x]:
                    cursor_with_outline.putpixel((new_x, new_y), (0, 0, 0, 255))
    
    # Combine outline and original cursor
    cursor_with_outline.alpha_composite(cursor)
    
    return np.array(cursor_with_outline)

def draw_cursor(frame, x, y, left_click=False, right_click=False, scaling_factor=1):
    """Draw a cursor on the frame at the given position"""
    # Convert coordinates to integers and apply scaling
    x, y = int(x * scaling_factor), int(y * scaling_factor)
    
    # Get cursor image
    cursor = get_cursor_image()
    
    # Calculate cursor placement bounds
    h, w = cursor.shape[:2]
    
    # Ensure cursor stays within frame bounds
    frame_h, frame_w = frame.shape[:2]
    x_start = max(0, x)
    y_start = max(0, y)
    x_end = min(frame_w, x + w)
    y_end = min(frame_h, y + h)
    
    # Calculate cursor image bounds
    cursor_x_start = max(0, -x)
    cursor_y_start = max(0, -y)
    cursor_x_end = cursor_x_start + (x_end - x_start)
    cursor_y_end = cursor_y_start + (y_end - y_start)
    
    # Only proceed if we have valid dimensions
    if x_end > x_start and y_end > y_start:
        # Get alpha channel
        alpha = cursor[cursor_y_start:cursor_y_end, cursor_x_start:cursor_x_end, 3] / 255.0
        alpha = alpha[..., np.newaxis]
        
        # Blend cursor with frame using alpha compositing
        cursor_part = cursor[cursor_y_start:cursor_y_end, cursor_x_start:cursor_x_end, :3]
        frame_part = frame[y_start:y_end, x_start:x_end]
        blended = (cursor_part * alpha + frame_part * (1 - alpha)).astype(np.uint8)
        frame[y_start:y_end, x_start:x_end] = blended
        
        # Optional: Add click indication
        if left_click or right_click:
            click_radius = 4
            click_color = (0, 255, 255) if left_click else (255, 255, 0)  # Cyan for left, yellow for right
            cv2.circle(frame, (x + 8, y + 8), click_radius, click_color, -1)
    
    return frame

async def record(save_dir: str = 'raw_data', save_name: str = 'record_0', 
                duration: int = 12, function_to_record: callable = None, 
                fn_args: tuple = ()):
    """
    Records mouse positions, clicks, and screen at 15 fps.
    Frames are stored at original resolution.
    """
    fps = 15
    interval = 1.0 / fps
    
    # Ensure directories exist with parent_directory
    os.makedirs(f'{parent_directory}/{save_dir}/videos', exist_ok=True)
    os.makedirs(f'{parent_directory}/{save_dir}/actions', exist_ok=True)

    # Initialize mouse data collection
    data = []
    
    # Start mouse listener
    listener = Listener(on_click=on_click)
    listener.start()

    with mss.mss(with_cursor=True) as sct:
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
        
        # Initialize video writer with original size
        codec = 'mp4v'
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(
            f'{parent_directory}/{save_dir}/videos/{save_name}.mp4', 
            fourcc, 
            fps, 
            (width, height),  # Use original size
            isColor=True
        )
           
        if not out.isOpened():
            raise Exception("Could not open video writer")
        
        print(f"Monitor settings: {monitor}")
        print(f"Recording at dimensions: width={width}, height={height}")
        
        frame_count = 0
        try:
            # Start function_to_record if provided
            if function_to_record:
                function_task = asyncio.create_task(asyncio.to_thread(function_to_record, *fn_args))
                await asyncio.sleep(0.1)  # Small delay to ensure function starts

            start_time = time.time()
            while time.time() - start_time < duration:
                frame_start = time.time()

                # Capture screen and mouse data
                screen = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
                
                # Draw cursor before writing
                x, y = mouse.position
                frame = draw_cursor(frame, x, y, left_click, right_click, scaling_factor)
                
                # Write frame at original size
                #frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
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
