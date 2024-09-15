
import obsws_python as obs
from record_mouse import record_mouse_actions

def script():
    # Connect to OBS
    ws = obs.ReqClient(host='192.168.1.177', port=4455, password='mypassword')  # Update with your WebSocket password if needed

    # Start OBS recording
    ws.start_record()

    # Start recording mouse actions
    max_x, max_y = record_mouse_actions(fps=15, duration=12)

    # Stop OBS recording
    ws.stop_record()

    # Disconnect from OBS
    ws.disconnect()

    print(f"Recorded mouse data on: width {max_x} height {max_y} screen.")

if __name__ == "__main__":
    script()
