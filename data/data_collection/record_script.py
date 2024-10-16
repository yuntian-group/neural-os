import obsws_python as obs
import os
import asyncio
from dotenv import load_dotenv
from data.data_collection.record_mouse import record_mouse_actions

# Load environment variables
load_dotenv()

current_directory = os.path.dirname(os.path.abspath(__file__)) #for obs video paths
parent_directory = os.path.dirname(current_directory)

async def record(save_dir: str = 'raw_data', save_name: str = 'record_0', function_to_record: callable = None, *fn_args: any):

    """
    Records mouse and video using obs. Saves it at save_path. REQUIRES obs save type to be .mp4! Change in obs settings!
    Paramters:
        function_to_record: optional function called when recording, used for synthetic data generation.
    """

    # Connect to OBS
    ws = obs.ReqClient(
        host=os.getenv("OBS_WS_HOST"),
        port=int(os.getenv("OBS_WS_PORT", 4455)),
        password=os.getenv("OBS_WS_PASSWORD")
    )
    ws.set_record_directory(parent_directory+r'/'+save_dir+r'/videos')
    ws.set_profile_parameter("Output","FilenameFormatting",fr'{save_name}')

    # Start OBS recording
    ws.start_record()

    # Start a task for function_to_record if provided
    if function_to_record:
        function_task = asyncio.create_task(asyncio.to_thread(function_to_record, *fn_args))

    # Record mouse actions for 1 second
    #max_x, max_y = record_mouse_actions(fps=15, duration=12, save_path=('../' + save_dir + '/actions/' + save_name + '.csv'))
    max_x, max_y = await asyncio.to_thread(
        record_mouse_actions, 
        fps=15, 
        duration=12, 
        save_path=('../' + save_dir + r'/actions/' + save_name + '.csv')
    )

    await function_task

    # Stop OBS recording after 1 second
    ws.stop_record()

    # If function_to_record was started, cancel it if it's still running
    if function_to_record:
        if not function_task.done():
            function_task.cancel()
            try:
                await function_task
            except asyncio.CancelledError:
                print("function_to_record was cancelled as it didn't complete within 1 second")

    # Disconnect from OBS
    ws.disconnect()

    print(f"Recorded mouse data on: width {max_x}px height {max_y}px screen.")
    print(f"Saved video and actions csv at {save_dir}/{save_name}.")
