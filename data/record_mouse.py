import pyautogui
import time
import pandas as pd
import numpy as np
from pynput.mouse import Controller, Listener, Button
from threading import Lock

mouse = Controller()
right_click = False
left_click = False

def on_click(x, y, button, pressed):

    global right_click
    global left_click
    if button == Button.right: right_click = pressed
    elif button == Button.left: left_click = pressed
    Button.x1

def record_mouse_actions_x(fps=12, duration=12, output_file='data/raw_data/mouse_actions.csv'):
    start_time = time.time()
    data = []
    interval = 1 / fps

    # Start the mouse listener in the background
    listener = Listener(on_click=on_click)
    listener.start()

    next_frame_time = start_time
    while time.time() - start_time < duration:
        current_time = time.time()
        
        if current_time >= next_frame_time:
            seconds = int(current_time - start_time)
            milliseconds = int((current_time - start_time - seconds) * 1000)
            time_formatted = f"{seconds}:{milliseconds}"

            x, y = mouse.position  # Get mouse position
            rclick = right_click
            lclick = left_click
            data.append([current_time - start_time, time_formatted, x, y, rclick, lclick])

            # Update the next frame time
            next_frame_time += interval

        # Optional: Sleep a small amount to prevent high CPU usage
        time.sleep(0.001) 

    # Stop the listener when done
    listener.stop()
    
    df = pd.DataFrame(data, columns=['Timestamp', 'Timestamp_formated', 'X', 'Y', 'Right Click', 'Left Click'])
    df.to_csv(output_file, index=False)

    return pyautogui.size()

def record_mouse_actions(fps=12, duration=12, output_file='data/raw_data/mouse_actions.csv'):
    start_time = time.time()
    data = []
    interval = 1 / fps

    elapsed_time = 0

    # Start the mouse listener in the background
    listener = Listener(on_click=on_click)
    listener.start()
    
    while elapsed_time < duration:
        current_time = time.time()

        seconds = int(current_time)
        milliseconds = int((current_time - seconds) * 1000)
        time_formatted = f"{seconds}:{milliseconds}"

        x, y = mouse.position  # Get mouse position
        rclick = right_click
        lclick = left_click
        data.append([current_time - start_time, time_formatted, x, y, rclick, lclick])
        
        # Sleep to maintain 15 fps
        time.sleep(interval)
        elapsed_time = current_time - start_time

    # Stop the listener when done
    listener.stop()
    
    df = pd.DataFrame(data, columns=['Timestamp', 'Timestamp_formated', 'X', 'Y', 'Right Click', 'Left Click'])
    df.to_csv(output_file, index=False)

    return pyautogui.size()

if __name__ == "__main__":
    record_mouse_actions(fps=12, duration=5, output_file='data/raw_data/mouse_actions.csv')