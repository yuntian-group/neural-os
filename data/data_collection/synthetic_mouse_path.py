import numpy as np
import time
from scipy.special import comb
import random


def bernstein_poly(i, n, t):
    return comb(n, i) * (t**(i)) * ((1-t)**(n-i))

def bezier_curve(points, num_points=1000):
    n = len(points) - 1
    t = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 2))
    for i, point in enumerate(points):
        curve += np.outer(bernstein_poly(i, n, t), point)
    return curve

def add_noise(curve, noise_level=0.1):
    noise = np.random.normal(0, noise_level, curve.shape)
    return curve + noise
# + int((1920 - 256)/2) + int((1080 - 256)/2)
def generate_control_points(num_points, screen_width, screen_height):
    """Generate control points with some points near or at the boundaries"""
    points = []
    for _ in range(num_points):
        if random.random() < 0.3:  # 30% chance of boundary point
            # Generate a point on or very close to a boundary
            if random.random() < 0.5:  # horizontal boundary
                x = random.randint(0, screen_width)
                y = random.choice([0, screen_height - 1]) if random.random() < 0.5 else random.randint(0, screen_height)
            else:  # vertical boundary
                x = random.choice([0, screen_width - 1]) if random.random() < 0.5 else random.randint(0, screen_width)
                y = random.randint(0, screen_height)
        else:
            # Regular point
            x = random.randint(0, screen_width)
            y = random.randint(0, screen_height)
        points.append((x, y))
    return points

def generate_human_like_trajectory(screen_width, screen_height,
                                   duration,  # Duration in seconds
                                   fps,  # Match recording FPS
                                   num_clicks,
                                   num_control_points=25, 
                                   double_click_prob=0.3):  # Probability of double click
    # Calculate number of points based on duration and fps
    num_points = int(duration * fps)
    HUMAN = random.random() > 0.5
    
    if HUMAN:
        # Generate more control points for more complex paths
        control_points = generate_control_points(num_control_points + random.randint(0, 10), 
                                              screen_width, screen_height)
        
        # Add some extra control points at the start and end to ensure full range
        if random.random() < 0.3:  # 30% chance of boundary-to-boundary movement
            # Add boundary points at start and end
            start_point = (random.choice([0, screen_width - 1]), random.randint(0, screen_height))
            end_point = (random.choice([0, screen_width - 1]), random.randint(0, screen_height))
            control_points = [start_point] + control_points + [end_point]
        
        curve = bezier_curve(control_points, num_points)
        noisy_curve = add_noise(curve, noise_level=0.15)  # Slightly increased noise
        trajectory = np.clip(noisy_curve, 0, [screen_width - 1, screen_height - 1]).astype(int)
    else: # just random points within the screen
        x = np.random.randint(0, screen_width, num_points)
        y = np.random.randint(0, screen_height, num_points)
        trajectory = np.vstack((x, y)).T
    
    # Generate fixed number of clicks at random points
    buffer = 0
    click_indices = np.random.choice(
        range(buffer, num_points - buffer), 
        size=num_clicks, 
        replace=False
    )
    click_indices.sort()  # Sort to maintain temporal order
    
    # Create click array with double clicks
    clicks = np.zeros(len(trajectory), dtype=bool)
    
    for idx in click_indices:
        if random.random() < double_click_prob and idx < len(trajectory) - 8:  # More room for second click
            # First click
            clicks[idx] = True
            
            # Gap calculation: 
            # At 24fps: 3 frames ≈ 125ms, 6 frames ≈ 250ms
            gap = random.randint(1, 6)  # Between 125ms and 250ms
            
            # Add small random movement between clicks
            #if not HUMAN:
            original_pos = trajectory[idx].copy()
            for j in range(1, gap + 1):
                jitter = np.random.normal(0, 1, 2)  # 1 pixel standard deviation for more stability
                new_pos = original_pos + jitter
                new_pos = np.clip(new_pos, 0, [screen_width - 1, screen_height - 1])
                trajectory[idx + j] = new_pos.astype(int)
        
            # Second click
            clicks[idx + gap] = True
        else:
            clicks[idx] = True  # Single click
    
    return list(zip(trajectory, clicks))

def generate_multiple_trajectories(num_trajectories, screen_width, screen_height, duration, fps):
    trajectories = []
    for _ in range(num_trajectories):
        # Randomly choose number of clicks for this trajectory
        num_clicks = np.random.randint(0, int(0.4*duration*fps))  # Random number of clicks proportional to duration
        trajectory = generate_human_like_trajectory(
            screen_width, screen_height,
            duration=duration,
            num_clicks=num_clicks,
            fps=fps
        )
        trajectories.append(trajectory)
    return trajectories

def move_mouse_through_trajectory(trajectory, delay=0.005):
    #screen_width, screen_height = pyautogui.size()
    for point in trajectory:
        x, y = point
        #pyautogui.moveTo(x + int((screen_width - 256)/2), y + int((screen_height - 256)/2))  # Move mouse to the point
        pyautogui.moveTo(x, y)  # Move mouse to the point

        # time.sleep(delay)

def plot_trajectories(trajectories, screen_width, screen_height):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    for trajectory in trajectories:
        x, y = trajectory.T
        #plt.plot(x + int((screen_width - 256)/2), y + int((screen_height - 256)/2))
        plt.plot(x, y)
        plt.scatter(x[0], y[0], color='green', s=50, label='Start')
        plt.scatter(x[-1], y[-1], color='red', s=50, label='End')
    plt.xlim(0, screen_width)
    plt.ylim(0, screen_height)
    plt.gca().invert_yaxis()  # Invert y-axis to match screen coordinates
    plt.title("Simulated Human-like Mouse Trajectories")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #trajectories = generate_multiple_trajectories(5, 256, 256)
    screen_width, screen_height = pyautogui.size()
    trajectories = generate_multiple_trajectories(5, screen_width, screen_height)
    
    # Print the trajectories
    for i, trajectory in enumerate(trajectories):
        print(f"Trajectory {i+1}:")
        for x, y in trajectory:
            print(f"({x}, {y})")
        print()
        move_mouse_through_trajectory(trajectory)
    
    # Plot the trajectories
    plot_trajectories(trajectories, screen_width, screen_height)
