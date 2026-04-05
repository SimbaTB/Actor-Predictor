import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue-purple, magenta, orange-yellow

# Data preparation
algorithms = ['Actor-Predictor', 'RSAC', 'VRM']
x_position = np.arange(len(algorithms))     # Bar positions
width = 0.35                                # Bar width

# CartPoleContinuous-v1: 
# Actor-Predictor: 44.2 min (0.737h)
# RSAC: 40.3 min (0.671h)
# VRM: 7.567 hour (7.742, 7.501, 7.518, 7.479, 7.596)

# Pendulum-v1:
# Actor-Predictor: 43.0 min (0.717h)
# RSAC: 37.6 min (0.627h)
# VRM: 7.857 hour (7.845, 7.891, 7.803, 7.865, 7.882)

# Training time data (hours)
task_x_time = [0.737, 0.671, 7.567]
task_y_time = [0.717, 0.627, 7.857]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)


# Task X bar chart
bars1 = ax1.bar(x_position, task_x_time, width, 
                color=colors, 
                edgecolor='black', 
                linewidth=1.2)

# Task Y bar chart
bars2 = ax2.bar(x_position, task_y_time, width, 
                color=colors, 
                edgecolor='black', 
                linewidth=1.2)

# Configure Task X subplot
ax1.set_title('CartPoleContinuous-v1', fontsize=14, fontweight='bold', pad=15)
# ax1.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average Training Time (hours)', fontsize=12, fontweight='bold')
ax1.set_xticks(x_position)
ax1.set_xticklabels(algorithms, rotation=0, ha='center', fontsize=11)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars for Task X
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
             f'{height:.2f} h', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Configure Task Y subplot
ax2.set_title('Pendulum-v1', fontsize=14, fontweight='bold', pad=15)
# ax2.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
ax2.set_xticks(x_position)
ax2.set_xticklabels(algorithms, rotation=0, ha='center', fontsize=11)
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars for Task Y
for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
             f'{height:.2f} h', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Adjust layout to prevent title cutoff
plt.tight_layout()

# Save the figure with extra space at the top
plt.savefig('./Visualization/training_time_comparison.png', dpi=300, bbox_inches='tight', 
            pad_inches=0.3, facecolor='white')

plt.show()