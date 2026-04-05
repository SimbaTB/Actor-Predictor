import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools
import agent
import envs

lr = 3e-4
alpha = 1
gamma = 0.99
tau = 0.005
hidden_dim = 128
grad_clip = 100
learning_starts = 1000
buffer_size = 1000_000
seq_len = 16
batch_size = 64
loss_scales = {"obs":1.0, "rew":1.0, "con":1.0, "actor":1.0, "critic":1.0, "alpha":0.1}

def plot(noised, pred, real, length):
    noised_x, noised_y = noised
    pred_x, pred_y = pred
    real_x, real_y = real
    
    fig = plt.figure(figsize=(16, 12))
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
    })
    
    t = np.arange(0, length)

    # ========== subfig 1 ========== 
    ax1 = fig.add_subplot(221, projection='3d')
    vmin = t.min() - 0.4*(t.max()-t.min())
    vmax = t.max() + 0.2*(t.max()-t.min())
    # 序列1: 低偏差、高方差
    sc1 = ax1.scatter(noised_x, noised_y, t, 
                    c=t, cmap='Blues', vmin=vmin, vmax=vmax,
                    s=30, alpha=0.7, 
                    label=f'Noised observation')

    # 序列2: 高偏差、低方差
    sc2 = ax1.scatter(pred_x, pred_y, t, 
                    c=t, cmap='Reds',  vmin=vmin, vmax=vmax,
                    s=30, alpha=0.7, 
                    label=f'Prediction')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Timestep')
    ax1.set_title('Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 添加颜色条表示时间
    # cbar = plt.colorbar(sc2, ax=ax1, pad=0.1)
    # cbar.set_label('Timestep')

    # ========== subfig 2 ========== 
    ax2 = fig.add_subplot(222)
    noised_x_centered = noised_x - real_x
    noised_y_centered = noised_y - real_y
    pred_x_centered = pred_x - real_x
    pred_y_centered = pred_y - real_y

    ax2.scatter(noised_x_centered, noised_y_centered, c='blue', s=20, alpha=0.3, label='Noised observation (centered)')
    ax2.scatter(pred_x_centered, pred_y_centered, c='red', s=20, alpha=0.3, label='Prediction (centered)')

    ellipse_noised = Ellipse(xy=(noised_x_centered.mean(), noised_y_centered.mean()), 
                         width=noised_x_centered.std(), height=noised_y_centered.std(),
                         edgecolor='blue', 
                         facecolor='none',
                         linewidth=2,
                         alpha=0.7,
                         label=f'Noised observation (±1σ)')
    ax2.add_patch(ellipse_noised)

    ellipse_pred = Ellipse(xy=(pred_x_centered.mean(), pred_y_centered.mean()), 
                         width=pred_x_centered.std(), height=pred_y_centered.std(),
                         edgecolor='red', 
                         facecolor='none',
                         linewidth=2,
                         alpha=0.7,
                         label=f'Prediction (±1σ)')
    ax2.add_patch(ellipse_pred)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Centered Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.6)

    # ========== subfig 3 ========== 
    ax3 = fig.add_subplot(223)
    ax3.scatter(t, noised_x, c='blue', s=20, alpha=0.6, label='Noised observation')
    ax3.scatter(t, pred_x, c='red', s=20, alpha=0.6, label='Prediction')
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('X')
    ax3.set_title('X - t')
    ax3.legend()
    ax3.grid(True, alpha=0.6)

    # ========== subfig 4 ========== 
    ax4 = fig.add_subplot(224)
    ax4.scatter(t, noised_y, c='blue', s=20, alpha=0.6, label='Noised observation')
    ax4.scatter(t, pred_y, c='red', s=20, alpha=0.6, label='Prediction')
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Y')
    ax4.set_title('Y - t')
    ax4.legend()
    ax4.grid(True, alpha=0.6)

    plt.tight_layout()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    args = parser.parse_args()

    tools.set_random_seed(args.seed)
    print(f"set random seed to {args.seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using device:", device)

    env_noise = envs.make_env("Pendulum-v1-P-N", seed=args.seed)
    env_pure = envs.make_env("Pendulum-v1-P", seed=args.seed)
    obs_space = env_noise.observation_space
    act_space = env_noise.action_space

    agent = agent.Agent(obs_space, hidden_dim, act_space, lr, grad_clip, alpha, tau, gamma, loss_scales, device)
    agent = torch.compile(agent)
    state_dict = torch.load(args.logdir + "/best_agent.pt", map_location=device)
    agent.load_state_dict(state_dict)

    noised, pred, real = [], [], []
    obs_noise, info = env_noise.reset()
    obs_pure, info = env_pure.reset()
    carry = agent.initial()
    reward, terminated, done = 0.0, False, False

    for i in range(200):
        action, carry = agent.take_action(obs_noise, carry, deterministic=False)
        with torch.no_grad():
            obs_pred, _, _, _ = agent.decoder(carry[0])
            obs_pred = obs_pred.squeeze().numpy()
        obs_noise, reward, terminated, truncated, info = env_noise.step(action)
        obs_pure, reward, terminated, truncated, info = env_pure.step(action)
        done = terminated or truncated
        
        if done:
            break

        if i > 0:
            noised.append(obs_noise[:2])
            pred.append(obs_pred[:2])
            real.append(obs_pure[:2])

    noised, pred, real = np.array(noised).T, np.array(pred).T, np.array(real).T
    print("noised.shape:", noised.shape)
    print("pred.shape:", pred.shape)
    print("real.shape:", real.shape)
    plot(noised, pred, real, noised.shape[1])
    figname = args.logdir + "/trajectory.png"
    plt.savefig(figname)
    print(f"Figure saved to {figname}")
    plt.show()