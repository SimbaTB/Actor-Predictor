import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'axes.facecolor': '#f8f9fa',
    'figure.facecolor': 'white',
})

colors = ['#4A90D9', '#E67E22', '#2ECC71']
algorithms = ['Actor-Predictor', 'RSAC', 'VRM']
x = np.arange(len(algorithms))
width = 0.35

task_x_time = [0.737, 0.671, 7.567]
task_y_time = [0.717, 0.627, 7.857]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), sharey=False)

def make_bars(ax, data, title):
    bars = ax.bar(x, data, width, color=colors, edgecolor='white', linewidth=1.5, alpha=0.9, capstyle='round')
    for bar, val in zip(bars, data):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08, f'{val:.2f}h', ha='center', va='bottom', fontsize=18, fontweight='bold', color='#2c3e50')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=20)
    ax.set_title(title, fontsize=25, fontweight='bold', pad=15, color='#2c3e50')
    ax.set_ylabel('Training Time (hours)', fontsize=20, fontweight='bold', color='#2c3e50')
    ax.set_ylim(0, max(data) * 1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.tick_params(colors='#555555', labelsize=20)
    ax.grid(axis='y', alpha=0.3, color='#cccccc', linestyle='--')
    ax.set_axisbelow(True)

make_bars(ax1, task_x_time, 'CartPoleContinuous-v1')
make_bars(ax2, task_y_time, 'Pendulum-v1')

#fig.suptitle('Training Time Comparison Across Algorithms', fontsize=26, fontweight='bold', color='#2c3e50', y=1.02)
plt.tight_layout()
plt.savefig('./Visualization/training_time_comparison.png', dpi=300, bbox_inches='tight', pad_inches=0.3, facecolor='white')
plt.show()
