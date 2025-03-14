import subprocess
import os
import time
from synthetic_mouse_path import generate_multiple_trajectories
import numpy as np
import multiprocessing
from functools import partial
import psutil
from tqdm import tqdm

SCREEN_WIDTH = 512
SCREEN_HEIGHT = 384

def initialize_clean_state():
    """Create and save a clean container state with initialized desktop"""
    print("Initializing clean container state...")
    
    # Start a container and let it initialize
    base_container_id = subprocess.check_output([
        'docker', 'run', '-d',
        '--hostname', 'neural-os',
        '--env', f'SCREEN_WIDTH={SCREEN_WIDTH}',
        '--env', f'SCREEN_HEIGHT={SCREEN_HEIGHT}',
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
    #import pdb; pdb.set_trace()
    # Convert numpy arrays to lists for JSON serialization
    #{'pos': array([48, 14]), 'left_click': False, 'right_click': False, 'key_events': {('keydown', 'browserback')}}
    def convert_trajectory(traj):
        return [
            ((int(d['pos'][0]), int(d['pos'][1])), True if d['left_click'] else False, True if d['right_click'] else False, list(d['key_events']))  # Use Python's True/False
            for d in traj
        ]
    
    # Convert and serialize
    trajectory_list = convert_trajectory(trajectory_data)
    #import pdb; pdb.set_trace()
    # Create a temporary Python script in the container
    script_content = f'''
import sys
import traceback

sys.path.append('/app')  # Add app directory to Python path

try:
    import os
    from record_script import record

    print("Python path:", sys.path)
    print("Contents of /app:", os.listdir('/app'))
    print("Current working directory:", os.getcwd())

    trajectory_data = {trajectory_list!r}  # Direct Python literal
    print("Starting recording with trajectory:", len(trajectory_data), "points")

    record(
        "raw_data",
        "record_{record_idx}",
        duration=30,
        fps=15,
        trajectory=trajectory_data
    )
except Exception as e:
    print("Error occurred:")
    traceback.print_exc()
    sys.exit(1)
'''
    #import pdb; pdb.set_trace()
    # Write the script to a temporary file in the container
    temp_script = f'/tmp/record_script_{record_idx}.py'
    cmd_write = [
        'docker', 'exec', container_id,
        'bash', '-c', f'cat > {temp_script} << \'EOL\'\n{script_content}\nEOL'
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

def process_trajectory(args, screen_width, screen_height, clean_state, memory_limit, max_retries=3):
    """Process a single trajectory in its own container with retries"""
    trajectory_idx, trajectory = args
    
    for attempt in range(max_retries):
        try:
            print(f"Recording trajectory {trajectory_idx} (attempt {attempt + 1}/{max_retries})")
            
            # Create a fresh container from clean state
            container_id = subprocess.check_output([
                'docker', 'run', '-d',
                '-v', f'{os.getcwd()}/raw_data:/app/raw_data',
                '--hostname', 'neural-os',
                '--env', 'DISPLAY=:99',
                '--env', f'SCREEN_WIDTH={screen_width}',
                '--env', f'SCREEN_HEIGHT={screen_height}',
                clean_state,
                '/app/start.sh'
            ]).decode().strip()
            
            success = False
            
            try:
                time.sleep(20)  # Wait for container to initialize
                record_trajectory(container_id, trajectory, trajectory_idx)
                success = True
            finally:
                # Always clean up the container
                subprocess.run(['docker', 'rm', '-f', container_id], check=False)
            
            if success:  # Only return after container cleanup
                return
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for trajectory {trajectory_idx}: {str(e)}")
            if attempt == max_retries - 1:  # Last attempt
                print(f"All {max_retries} attempts failed for trajectory {trajectory_idx}")
                print (e)  # Re-raise the last exception
                print ('Ignoring this error')
                print ('*'*100)
            time.sleep(2)  # Wait a bit before retrying

def create_synthetic_dataset(n=1, max_workers=None, memory_per_worker='2g'):
    """Create synthetic dataset with resource management"""
    screen_width = SCREEN_WIDTH
    screen_height = SCREEN_HEIGHT
    
    # Calculate optimal number of workers based on system resources
    total_memory_gb = psutil.virtual_memory().total / (1024**3)  # Convert to GB
    num_cpus = os.cpu_count()
    
    if max_workers is None:
        # Calculate based on available resources
        memory_needed_per_worker = 2  # GB
        max_by_memory = int((total_memory_gb - 4) / memory_needed_per_worker)  # Leave 4GB for system
        max_by_cpu = num_cpus - 2  # Leave 2 CPU cores for system
        max_workers = min(max_by_memory, max_by_cpu)
        
        # For very large machines, maybe cap at 32 or adjust based on your needs
        max_workers = min(max_workers, 64)  # Optional cap
    
    print(f"System resources: {num_cpus} CPUs, {total_memory_gb:.1f}GB RAM")
    print(f"Using {max_workers} workers")
    
    # Initialize clean state first
    clean_state = initialize_clean_state()
    
    try:
        # Generate all trajectories first with progress bar
        print("Generating all trajectories...")
        trajectories = list(tqdm(
            generate_multiple_trajectories(n, screen_width, screen_height, duration=30, fps=15),
            total=n,
            desc="Generating trajectories"
        ))
        #import pdb; pdb.set_trace()
        
        # Process in parallel with resource limits and progress bar
        process_func = partial(
            process_trajectory,
            screen_width=screen_width,
            screen_height=screen_height,
            clean_state=clean_state,
            memory_limit=memory_per_worker
        )
        
        with multiprocessing.Pool(max_workers) as pool:
            list(tqdm(
                pool.imap(process_func, enumerate(trajectories)),
                total=len(trajectories),
                desc="Processing trajectories"
            ))
    
    finally:
        # Cleanup
        subprocess.run(['docker', 'rmi', clean_state], check=True)
        subprocess.run('docker container prune -f', shell=True, check=False)

if __name__ == "__main__":
    # Ensure raw_data directory exists
    os.makedirs('raw_data', exist_ok=True)
    
    # Set higher recursion limit for multiprocessing
    import sys
    sys.setrecursionlimit(10000)
    
    # Run with parallel processing
    #create_synthetic_dataset(16000)
    create_synthetic_dataset(24000)