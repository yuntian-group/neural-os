import subprocess
import os
import time
from synthetic_mouse_path import generate_multiple_trajectories
import random
import numpy as np
import json

def initialize_clean_state():
    """Create and save a clean container state with initialized desktop"""
    print("Initializing clean container state...")
    
    # Start a container and let it initialize
    base_container_id = subprocess.check_output([
        'docker', 'run', '-d',
        '--env', f'SCREEN_WIDTH=1024',
        '--env', f'SCREEN_HEIGHT=768',
        'synthetic_data_generator',
        '/app/start.sh'
    ]).decode().strip()
    
    # Wait for XFCE to fully initialize
    time.sleep(5)
    
    # Save the clean state
    clean_state = subprocess.check_output([
        'docker', 'commit', base_container_id
    ]).decode().strip()
    
    # Clean up the initialization container
    subprocess.run(['docker', 'rm', '-f', base_container_id], check=True)
    
    return clean_state

def record_trajectory(container_id, trajectory_data, record_idx):
    """Send trajectory data to container and record"""
    # Convert numpy arrays to lists for JSON serialization
    def convert_trajectory(traj):
        return [
            ((int(x), int(y)), True if click else False)  # Use Python's True/False
            for ((x, y), click) in traj
        ]
    
    # Convert and serialize
    trajectory_list = convert_trajectory(trajectory_data)
    
    # Create a temporary Python script in the container
    script_content = f'''
import sys
import traceback

sys.path.append('/app')  # Add app directory to Python path

try:
    import json
    import os
    from record_script import record

    print("Python path:", sys.path)
    print("Contents of /app:", os.listdir('/app'))
    print("Current working directory:", os.getcwd())
    
    trajectory_data = {trajectory_list}  # Direct Python literal
    print("Starting recording with trajectory:", len(trajectory_data), "points")
    
    # Create output directories if they don't exist
    os.makedirs("raw_data/videos", exist_ok=True)
    os.makedirs("raw_data/actions", exist_ok=True)
    
    record(
        "raw_data",
        "record_{record_idx}",
        duration=120,
        trajectory=trajectory_data
    )
except Exception as e:
    print("Error occurred:")
    traceback.print_exc()
    sys.exit(1)
'''
    
    # Write the script to a temporary file in the container
    temp_script = f'/tmp/record_script_{record_idx}.py'
    cmd_write = [
        'docker', 'exec', container_id,
        'bash', '-c', f'cat > {temp_script} << EOL\n{script_content}\nEOL'
    ]
    subprocess.run(cmd_write, check=True)
    
    # Execute the script with proper Python path
    cmd_execute = [
        'docker', 'exec',
        '-e', 'DISPLAY=:99',
        '-e', 'PYTHONPATH=/app',
        container_id,
        'bash', '-c',
        f'''
set -x  # Print commands as they execute
export PYTHONUNBUFFERED=1  # Ensure Python output isn't buffered

# Debug info
echo "Current environment:"
env | grep DISPLAY
env | grep PYTHON
echo "Contents of /app:"
ls -la /app/
echo "X server status:"
xdpyinfo | head -n 5 || echo "X server not running"
echo "Process status:"
ps aux | grep X

# Run the script with full error output
cd /app  # Change to app directory
python3 -u {temp_script}
'''
    ]
    
    # Run with output capture
    result = subprocess.run(cmd_execute, capture_output=True, text=True)
    print("Command output:")
    print(result.stdout)
    print("Error output:")
    print(result.stderr)
    result.check_returncode()

def create_synthetic_dataset(n=1):
    """Create synthetic dataset using container state restoration"""
    screen_width = 1024
    screen_height = 768
    
    # Create clean state with properly initialized X server
    print("Initializing clean container state...")
    
    # Start a container and let it initialize
    base_container_id = subprocess.check_output([
        'docker', 'run', '-d',
        '--env', 'DISPLAY=:99',
        '--env', f'SCREEN_WIDTH={screen_width}',
        '--env', f'SCREEN_HEIGHT={screen_height}',
        'synthetic_data_generator',
        '/app/start.sh'
    ]).decode().strip()
    
    # Wait for X server to be ready
    time.sleep(5)
    
    # Test X server
    subprocess.run([
        'docker', 'exec',
        base_container_id,
        'bash', '-c',
        '''
        for i in $(seq 1 10); do 
            if xdpyinfo >/dev/null 2>&1; then 
                echo "X server is ready"
                exit 0
            fi
            sleep 1
        done
        echo "X server failed to start"
        exit 1
        '''
    ], check=True)
    
    # Save the clean state
    clean_state = subprocess.check_output([
        'docker', 'commit', base_container_id
    ]).decode().strip()
    
    # Clean up the initialization container
    subprocess.run(['docker', 'rm', '-f', base_container_id], check=True)
    
    try:
        # Generate all trajectories first
        print("Generating all trajectories...")
        trajectories = generate_multiple_trajectories(n, screen_width, screen_height)
        
        # Process each trajectory
        for i, trajectory in enumerate(trajectories):
            print(f"Recording trajectory {i}/{n}")
            
            # Create a fresh container from clean state
            container_id = subprocess.check_output([
                'docker', 'run', '-d',
                '-v', f'{os.getcwd()}/raw_data:/app/raw_data',
                '--env', 'DISPLAY=:99',
                '--env', f'SCREEN_WIDTH={screen_width}',
                '--env', f'SCREEN_HEIGHT={screen_height}',
                clean_state,
                '/app/start.sh'
            ]).decode().strip()
            
            try:
                # Wait longer for desktop to be fully ready
                time.sleep(10)  # Increased from 2 to 3 seconds
                record_trajectory(container_id, trajectory, i)
            finally:
                subprocess.run(['docker', 'rm', '-f', container_id])
            
            # Optional: Small delay between recordings
            time.sleep(0.1)
    
    finally:
        # Clean up the saved state
        subprocess.run(['docker', 'rmi', clean_state], check=True)

if __name__ == "__main__":
    # Ensure raw_data directory exists
    os.makedirs('raw_data', exist_ok=True)
    os.makedirs('raw_data/videos', exist_ok=True)
    os.makedirs('raw_data/actions', exist_ok=True)
    
    # Run without batching
    create_synthetic_dataset(4)
