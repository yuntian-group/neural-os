import subprocess
import time

def start_obs():
    # Start OBS (Adjust path to your OBS executable)
    subprocess.Popen([r"C:\\Program Files (x86)\\obs-studio\\bin\\64bit\\obs64.exe", "--startrecording"], cwd=r"C:\Program Files (x86)\obs-studio\obs-plugins\64bit")

def record_mouse_actions():
    # Import the mouse recording function from your script
    from record_mouse import record_mouse_actions
    record_mouse_actions(fps=12, duration=12)

def script():
    # Start OBS
    start_obs()

    # Start recording mouse actions
    record_mouse_actions()
    print("We are recording!")

if __name__ == "__main__":
    script()
