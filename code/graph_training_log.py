import re
import numpy as np
import matplotlib.pyplot as plt

def read_training_log(file_path):
    """
    Reads the training log file and extracts all rewards in sequential order across all checkpoints.
    Assumes the log file lists checkpoints in order, with episodes numbered relative to each checkpoint.
    """
    all_rewards = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Episode"):
                # Match episode line format: "Episode: <num>, Total Reward: <value>"
                m = re.search(r'Episode:\s*(\d+),\s*Total Reward:\s*([-+]?[0-9]*\.?[0-9]+)', line)
                if m:
                    reward = float(m.group(2))
                    all_rewards.append(reward)
    return all_rewards

def plot_training_curve(all_rewards):
    """
    Plots the 50-point moving average of the rewards across all episodes.
    X-axis is the episode number at the end of each 50-episode window.
    """
    window = 50
    if len(all_rewards) < window:
        print(f"Not enough data to compute {window}-point moving average.")
        return

    # Compute 50-point moving averages
    averages = []
    for i in range(len(all_rewards) - window + 1):
        avg = np.mean(all_rewards[i:i + window])
        averages.append(avg)

    # X-values are episode numbers at the end of each window (e.g., 50, 51, ...)
    x_values = np.arange(window, len(all_rewards) + 1)

    # Plot the moving average as a line
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, averages, label=f'{window}-point moving average', color='blue')
    plt.xlabel('Episode Number')
    plt.ylabel('Average Total Reward')
    plt.title('Training Curve (50-point Moving Average)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    file_path = 'training_log.txt'
    all_rewards = read_training_log(file_path)
    plot_training_curve(all_rewards)