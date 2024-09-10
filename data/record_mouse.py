import pyautogui
import time
import pandas as pd
from pynput.mouse import Controller, Listener

mouse = Controller()

# Variable to track if the mouse button is pressed
mouse_pressed = False

# Callback function to detect when mouse button is pressed
def on_click(x, y, button, pressed):
    global mouse_pressed
    mouse_pressed = pressed  # Update the mouse_pressed state

def record_mouse_actions(fps=12, duration=12, output_file='data/raw_data/mouse_actions.csv'):
    start_time = time.time()
    data = []
    interval = 1 / fps

    # Start the mouse listener in the background
    listener = Listener(on_click=on_click)
    listener.start()
    
    while time.time() - start_time < duration:
        current_time = time.time() - start_time

        seconds = int(current_time)
        milliseconds = int((current_time - seconds) * 1000)
        time_formatted = f"{seconds}:{milliseconds}"

        x, y = mouse.position  # Get mouse position
        clicked = mouse_pressed  # Get mouse click status
        data.append([time_formatted, x, y, clicked])
        
        # Wait for the next frame
        time.sleep(interval)

    # Stop the listener when done
    listener.stop()
    
    df = pd.DataFrame(data, columns=['Timestamp', 'X', 'Y', 'Clicked'])
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    record_mouse_actions(fps=12, duration=5, output_file='data/raw_data/mouse_actions.csv')