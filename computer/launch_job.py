import subprocess
import time

CHECK_INTERVAL = 10  # seconds
CONSECUTIVE_ZEROS = 10
SCRIPT_COMMAND = "python cluster_prev_curr_transitions.py"

zero_usage_counter = 0

while True:
    try:
        # Run nvidia-smi to get GPU usage percentage
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpu_usages = [int(x) for x in result.stdout.strip().split('\n')]
        max_gpu_usage = max(gpu_usages)

        if max_gpu_usage == 0:
            zero_usage_counter += 1
            print(f"GPU usage is 0%. ({zero_usage_counter}/{CONSECUTIVE_ZEROS})")
        else:
            zero_usage_counter = 0
            print(f"GPU usage: {gpu_usages}%")

        if zero_usage_counter >= CONSECUTIVE_ZEROS:
            print("GPU has been idle for consecutive checks. Running the script...")
            subprocess.run(SCRIPT_COMMAND, shell=True)
            break

    except subprocess.CalledProcessError as e:
        print("Error checking GPU usage:", e)
        break

    time.sleep(CHECK_INTERVAL)

