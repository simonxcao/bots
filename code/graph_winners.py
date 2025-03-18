import re
import numpy as np
import matplotlib.pyplot as plt

def read_training_log(file_path):
    """
    Reads the training log file and extracts data for only the last two checkpoints.
    Checkpoint 4 starts at episode 1501.
    Checkpoint 5 starts 750 episodes later (at 2251).
    Individual points are plotted in the sequential order they appear in the file.
    """
    # Fixed starting points for checkpoints 4 and 5
    checkpoint_base = {4: 1501, 5: 2251}
    
    # Store data for all checkpoints
    all_checkpoints_data = []
    
    with open(file_path, 'r') as f:
        current_checkpoint = None
        current_data = None
        
        for line in f:
            line = line.strip()
            if line.startswith("Checkpoint"):
                match = re.search(r'Checkpoint\s+(\d+)', line)
                if match:
                    current_checkpoint = int(match.group(1))
                    if current_checkpoint in [4, 5]:  # Only process the last two checkpoints
                        current_data = {
                            'checkpoint': current_checkpoint,
                            'episodes': [],
                            'rewards': [],
                            'base': checkpoint_base.get(current_checkpoint)
                        }
                        all_checkpoints_data.append(current_data)
            elif line.startswith("Episode") and current_data is not None:
                m = re.search(r'Episode:\s*(\d+),\s*Total Reward:\s*([-+]?[0-9]*\.?[0-9]+)', line)
                if m:
                    episode_num = int(m.group(1))
                    reward = float(m.group(2))
                    
                    # Calculate absolute episode number using the fixed base
                    absolute_episode = current_data['base'] + episode_num
                    
                    current_data['episodes'].append(absolute_episode)
                    current_data['rewards'].append(reward)
    
    # Extract data in sequential order
    x_points = []
    y_points = []
    checkpoint_indices = {}
    checkpoint_numbers = []
    
    for data in all_checkpoints_data:
        checkpoint = data['checkpoint']
        start_idx = len(x_points)
        x_points.extend(data['episodes'])
        y_points.extend(data['rewards'])
        end_idx = len(x_points)
        checkpoint_indices[checkpoint] = (start_idx, end_idx)
        checkpoint_numbers.append(checkpoint)
    
    return x_points, y_points, checkpoint_indices, checkpoint_numbers

def plot_training_curve(x, y, checkpoint_indices, checkpoint_numbers):
    """
    Plots individual points for checkpoints 4 and 5.
    The x-axis represents the absolute episode number, and the y-axis is the reward.
    Uses a third-degree polynomial for a smoother fit.
    """
    plt.figure(figsize=(10, 6))
    
    # Use different colors for different checkpoints
    colors = ['blue', 'green']
    
    for i, checkpoint in enumerate(checkpoint_numbers):
        start_idx, end_idx = checkpoint_indices[checkpoint]
        plt.scatter(
            x[start_idx:end_idx], 
            y[start_idx:end_idx], 
            color=colors[i], 
            label=f'Checkpoint {checkpoint}',
            alpha=0.7,
            s=30
        )
    
    # Fit a 3rd degree polynomial for a smoother curve
    if len(x) > 3:  # Make sure we have enough points for a 3rd degree polynomial
        degree = 3
        coefficients = np.polyfit(x, y, degree)
        poly_func = np.poly1d(coefficients)
        
        x_fit = np.linspace(min(x), max(x), 200)
        y_fit = poly_func(x_fit)
        plt.plot(x_fit, y_fit, color='red', label=f'{degree}rd-degree best-fit curve')
    
    plt.xlabel('Episode Number')
    plt.ylabel('Total Reward')
    plt.title('Training Curve (Individual Points)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    file_path = 'training_log.txt'
    x_points, y_points, checkpoint_indices, checkpoint_numbers = read_training_log(file_path)
    plot_training_curve(x_points, y_points, checkpoint_indices, checkpoint_numbers)