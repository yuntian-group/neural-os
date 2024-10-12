import numpy as np
import pyautogui
import time
import matplotlib.pyplot as plt
from scipy.special import comb

def bernstein_poly(i, n, t):
    return comb(n, i) * (t**(i)) * ((1-t)**(n-i))

def bezier_curve(points, num_points=1000):
    n = len(points) - 1
    t = np.linspace(0, 1, num_points) ** 1.1
    curve = np.zeros((num_points, 2))
    for i, point in enumerate(points):
        curve += np.outer(bernstein_poly(i, n, t), point)
    return curve

def add_noise(curve, noise_level=0.1):
    noise = np.random.normal(0, noise_level, curve.shape)
    return curve + noise
# + int((1920 - 256)/2) + int((1080 - 256)/2)
def generate_control_points(num_points, screen_width, screen_height):
    x = np.random.randint(0, screen_width, num_points)
    y = np.random.randint(0, screen_height, num_points)
    return list(zip(x, y))

def generate_human_like_trajectory(screen_width=1920, screen_height=1080, 
                                   num_control_points=5, num_points=1000):
    control_points = generate_control_points(num_control_points, screen_width, screen_height)
    curve = bezier_curve(control_points, num_points)
    noisy_curve = add_noise(curve)
    return np.clip(noisy_curve, 0, [screen_width, screen_height]).astype(int)

def generate_multiple_trajectories(num_trajectories=5, screen_width=1920, screen_height=1080):
    trajectories = []
    for _ in range(num_trajectories):
        trajectory = generate_human_like_trajectory(screen_width, screen_height)
        trajectories.append(trajectory)
    return trajectories

def move_mouse_through_trajectory(trajectory, delay=0.005):
    for point in trajectory:
        x, y = point
        pyautogui.moveTo(x + int((1920 - 256)/2), y + int((1080 - 256)/2))  # Move mouse to the point
        # time.sleep(delay)

def plot_trajectories(trajectories, screen_width=1920, screen_height=1080):
    plt.figure(figsize=(12, 6))
    for trajectory in trajectories:
        x, y = trajectory.T
        plt.plot(x + int((1920 - 256)/2), y + int((1080 - 256)/2))
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
    trajectories = generate_multiple_trajectories(5, 256, 256)
    
    # Print the trajectories
    for i, trajectory in enumerate(trajectories):
        print(f"Trajectory {i+1}:")
        for x, y in trajectory:
            print(f"({x}, {y})")
        print()
    
    # Plot the trajectories
    plot_trajectories(trajectories)