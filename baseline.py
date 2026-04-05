import os
os.environ["MUJOCO_GL"] = "egl"
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import RecurrentPPO
import argparse
import envs
import tools
import time

class ReportCallback(BaseCallback):
    def __init__(self, eval_env, report_every, logdir, recurrent=False, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.report_every = report_every
        self.logdir = logdir
        self.recurrent = recurrent
        self.last_step = self.num_timesteps
        self.last_time = time.time()
        
    def _on_step(self):
        if self.num_timesteps % self.report_every == 0:
            metrics = {}
            returns, lengths = evaluate_policy(self.model, self.eval_env, n_eval_episodes=5, return_episode_rewards=True)
            metrics["eval_return"] = np.mean(returns).item()
            metrics["eval_std"] = np.std(returns).item()
            metrics["eval_steps"] = np.mean(lengths).item()
            current_time = time.time()
            metrics["fps"] = (self.num_timesteps - self.last_step) / (current_time - self.last_time)
            self.last_step, self.last_time = self.num_timesteps, current_time

            for key, value in metrics.items():
                self.logger.record(key, value)
            
            metrics["step"] = self.num_timesteps
            metrics["eval_returns(raw)"] = returns
            log = f"[steps {self.num_timesteps/1000}k] {metrics}\n"

            if self.verbose > 0:
                print(log)
            
            with open(logdir + "/metrics.txt", "a") as file:
                file.write(log)
        return True

parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", default="PPO", type=str, choices=["PPO", "SAC", "TD3", "RecurrentPPO"], 
                    help="Algorithm name(SAC or PPO)")
parser.add_argument("--logdir", help="Directory to save log")
parser.add_argument("--task", default="BipedalWalker-v3", type=str, help="Gymnasium environment id")
parser.add_argument("--steps", default=1000000, type=int, help="Max number of steps that the agent interacts with the environment")
parser.add_argument("--seed", default=0, type=int, help="Random seed")
args = parser.parse_args()

tools.set_random_seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device:", device)

logdir = args.logdir or f"./runs/{args.task.replace('ALE/', '')}_{args.algorithm}_{str(time.time())[-5:]}"
print("logdir:", logdir)

train_env = Monitor(envs.make_env(args.task, seed=args.seed))
eval_env = Monitor(envs.make_env(args.task, seed=args.seed))
print("obs_space:", train_env.observation_space)
print("act_space:", train_env.action_space)

policy_type = "CnnPolicy" if tools.is_image(train_env.observation_space) else "MlpPolicy"
rec_policy_type = "CnnLstmPolicy" if tools.is_image(train_env.observation_space) else "MlpLstmPolicy"

policy_kwargs = {
    "net_arch": [128, 128],
    "activation_fn": nn.SiLU, 
    "optimizer_class": torch.optim.Adam,
    "normalize_images": False,
}
rec_policy_kwargs = {
    "lstm_hidden_size": 128,
    "n_lstm_layers": 3,
    "activation_fn": nn.SiLU,
    "optimizer_class": torch.optim.Adam,
    "normalize_images": False,
}

if args.algorithm == "SAC":
    recurrent = False
    model = SAC(policy_type, train_env, verbose=0, tensorboard_log=logdir, device=device, policy_kwargs=policy_kwargs, 
             learning_rate=3e-4, learning_starts=1000, buffer_size=1000000, train_freq=1, batch_size=256)
elif args.algorithm == "TD3":
    recurrent = False
    model = TD3(policy_type, train_env, verbose=0, tensorboard_log=logdir, device=device, policy_kwargs=policy_kwargs,
             learning_rate=3e-4, learning_starts=1000, buffer_size=1000000, train_freq=1, batch_size=256, 
             policy_delay=2, target_policy_noise=0.2, target_noise_clip=0.5)
elif args.algorithm == "PPO":
    recurrent = False
    model = PPO(policy_type, train_env, verbose=0, tensorboard_log=logdir, device=device, policy_kwargs=policy_kwargs, 
             learning_rate=3e-4, n_steps=4000, batch_size=32, n_epochs=10, clip_range=0.2)
elif args.algorithm == "RecurrentPPO":
    recurrent = True
    model = RecurrentPPO(rec_policy_type, train_env, verbose=0, tensorboard_log=logdir, policy_kwargs=rec_policy_kwargs,
             learning_rate=3e-4, n_steps=200, batch_size=100, n_epochs=10, clip_range=0.2)
else:
    raise ValueError(f"invalid algorithm:{args.algorithm}")

# print(model.policy)

callback = ReportCallback(eval_env, 4000, logdir, recurrent)
model.learn(total_timesteps=args.steps, callback=callback)