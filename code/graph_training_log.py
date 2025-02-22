import re
import numpy as np
import matplotlib.pyplot as plt

def read_training_log(file_path):
    """
    Reads the training log file and computes 25-episode averages.
    Each checkpoint’s recorded episodes are converted to cumulative episodes 
    by adding an offset based on the total training episodes:
        - Checkpoint 0: offset = 0
        - Checkpoint 1: offset = 950
        - Checkpoint 2: offset = 1600
        - Checkpoint 3: offset = 2550
        - Checkpoint 4: offset = 3450

    For every 25 episodes in a checkpoint, the average total reward is computed.
    The average is plotted at the midpoint of the 25-episode block (after applying the offset).
    """
    checkpoint_offsets = {0: 0, 1: 950, 2: 1600, 3: 2550, 4: 3450}
    
    x_points = []  # cumulative episode midpoints
    y_points = []  # corresponding average rewards

    with open(file_path, 'r') as f:
        current_checkpoint = None
        current_offset = 0
        current_rewards = []
        current_episode_numbers = []
        
        for line in f:
            line = line.strip()
            if line.startswith("Checkpoint"):
                match = re.search(r'Checkpoint\s+(\d+)', line)
                if match:
                    current_checkpoint = int(match.group(1))
                    current_offset = checkpoint_offsets.get(current_checkpoint, 0)
                    current_rewards = []
                    current_episode_numbers = []
            elif line.startswith("Episode"):
                m = re.search(r'Episode:\s*(\d+),\s*Total Reward:\s*([-+]?[0-9]*\.?[0-9]+)', line)
                if m:
                    episode_num = int(m.group(1))
                    reward = float(m.group(2))
                    current_rewards.append(reward)
                    current_episode_numbers.append(episode_num)
                    
                    if len(current_rewards) == 25:
                        avg_reward = sum(current_rewards) / 25.0
                        first = current_episode_numbers[0]
                        last = current_episode_numbers[-1]
                        mid_recorded = (first + last) / 2.0
                        cumulative_ep = current_offset + mid_recorded
                        x_points.append(cumulative_ep)
                        y_points.append(avg_reward)
                        current_rewards = []
                        current_episode_numbers = []
    return x_points, y_points

def plot_training_curve(x, y):
    """
    Plots the 25-episode averages along with a smooth best-fit curve.
    The x-axis represents the cumulative episode number, and the y-axis is the reward.
    Uses a third-degree polynomial for a smoother fit.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='25-episode average')
    
    # Fit a 3rd degree polynomial for a smoother curve
    degree = 3
    coefficients = np.polyfit(x, y, degree)
    poly_func = np.poly1d(coefficients)
    
    x_fit = np.linspace(min(x), max(x), 200)
    y_fit = poly_func(x_fit)
    plt.plot(x_fit, y_fit, color='red', label=f'{degree}rd-degree best-fit curve')
    
    plt.xlabel('Cumulative Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Curve (25-episode Averages)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    file_path = 'training_log.txt'
    x_points, y_points = read_training_log(file_path)
    plot_training_curve(x_points, y_points)
