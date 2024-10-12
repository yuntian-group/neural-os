
import obsws_python as obs
import os
import asyncio
from data.data_collection.record_mouse import record_mouse_actions

current_directory = os.path.dirname(os.path.abspath(__file__)) #for obs video paths
parent_directory = os.path.dirname(current_directory)

async def record(save_dir: str = 'raw_data', save_name: str = 'record_0', function_to_record: callable = None, *fn_args: any):

    """
    Records mouse and video using obs. Saves it at save_path. REQUIRES obs save type to be .mp4! Change in obs settings!
    Paramters:
        function_to_record: optional function called when recording, used for synthetic data generation.
    """

    # Connect to OBS
    ws = obs.ReqClient(host=os.getenv("ip"), port=4455, password='mypassword')  # Update with your WebSocket password if needed
    ws.set_record_directory(parent_directory+r'/'+save_dir+r'/')
    ws.set_profile_parameter("Output","FilenameFormatting",fr'{save_name}')


    # Start OBS recording
    ws.start_record()

    if function_to_record:
        await asyncio.to_thread(function_to_record, *fn_args)

    # Start recording mouse actions and saves the csv
    max_x, max_y = record_mouse_actions(fps=15, duration=12, save_path=('../' + save_dir + '/' + save_name + '.csv'))

    # Stop OBS recording
    ws.stop_record()

    # Disconnect from OBS
    ws.disconnect()

    print(f"Recorded mouse data on: width {max_x}px height {max_y}px screen.")
    print(f"Saved video and actions csv at {save_dir}/{save_name}.")

