from record_script import record
import asyncio
import time
import pyautogui
from synthetic_mouse_path import generate_multiple_trajectories, move_mouse_through_trajectory
import random
import numpy as np

random.seed(1234)
np.random.seed(1234)

async def create_synthetic_dataset(n=1):
    screen_width, screen_height = pyautogui.size()
    trajectories = generate_multiple_trajectories(n, screen_width, screen_height)

    for i, trajectory in enumerate(trajectories):
        await record('raw_data', f'record_{i}', duration=12, 
                    function_to_record=move_mouse_through_trajectory, 
                    fn_args=(trajectory,))
        time.sleep(0.5)

if __name__ == "__main__":
    asyncio.run(create_synthetic_dataset(4000))
