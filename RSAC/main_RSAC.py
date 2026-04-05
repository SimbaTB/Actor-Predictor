import os
os.environ["MUJOCO_GL"] = "egl"
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
import argparse
import time
import multiprocessing
import agent_RSAC
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import replay
import tools
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

def train_policy(logdir, step, train_env, eval_env, train_agent, eval_agent, replay_buffer, 
                 total_steps, batch_size=batch_size, 
                 learning_starts=learning_starts, update_every=1, report_every=4000, save=False, use_async_eval=False):
    step = 0
    episode = 0
    
    last_step, last_time = 0, time.time()
    report_process = None

    metrics = {}
    while step < total_steps:
        episode += 1
        obs, info = train_env.reset()
        carry = train_agent.initial()
        reward, terminated, done = 0.0, False, False

        while not done and step < total_steps:
            if step < learning_starts:
                action = train_env.action_space.sample()
            else:
                action, carry = train_agent.take_action(obs, carry, deterministic=False)
            replay_buffer.add(obs, action, reward, terminated)
            obs, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated
            step += 1

            if replay_buffer.get_size() > learning_starts and step % update_every == 0:
                batch = replay_buffer.sample(batch_size)
                if batch is not None:
                    train_agent.train_step(**batch)
            
            if step % report_every == 0 or step == total_steps:
                if save:
                    save_checkpoint(logdir, train_agent, replay_buffer, step)
                
                fps = (step - last_step) / (time.time() - last_time)
                last_step, last_time = step, time.time()
                metrics = train_agent.report()
                eval_agent.load_state_dict(train_agent.state_dict())
                
                if use_async_eval:
                    # 等待上一次测评结束
                    if report_process:
                        report_process.join()

                    batch = replay_buffer.sample(6, device="cpu")
                    report_process = multiprocessing.Process(
                        target=report,
                        args=(logdir, metrics, eval_env, eval_agent, batch, step, episode, fps)
                    )
                    report_process.start()
                else:
                    batch = replay_buffer.sample(6)
                    report(logdir, metrics, eval_env, eval_agent, batch, step, episode, fps)

        # 添加最后一个时间步的结果
        action, carry = train_agent.take_action(obs, carry, deterministic=False)
        replay_buffer.add(obs, action, reward, terminated)

    # 等待最后一次测评结束
    if use_async_eval and report_process:
        report_process.join()

# 支持异步测试，不阻塞训练过程
def report(logdir, metrics, eval_env, eval_agent, batch, step, episode, fps):
    metrics["step"], metrics["episode"] = step, episode
    writer = SummaryWriter(log_dir=logdir)
    eval_return, eval_steps, video = evaluate_policy(eval_env, eval_agent)
    metrics["eval_steps"] = eval_steps
    metrics["eval_return"] = eval_return.mean().item()
    metrics["eval_std"] = eval_return.std().item()
    writer.add_video("eval_video", video, global_step=step, fps=16)
    metrics["fps"] = fps

    for name, value in metrics.items():
        writer.add_scalar(name, value, step)
    
    metrics["eval_return(raw)"] = eval_return.tolist()
    log = f"[step {step/1000}k, episode {episode}]" + str(metrics) + "\n"
    print(log)

    with open(logdir + "/metrics.txt", "a") as file:
        file.write(log)
    
    writer.close()

def evaluate_policy(env, agent, num_episodes = 5):
    returns = np.zeros((num_episodes,), dtype=np.float32)
    total_steps = 0
    frames = []

    for i in range(num_episodes):
        obs, info = env.reset()
        carry = agent.initial()
        reward, terminated, done = 0.0, False, False
        eval_steps = 0

        while not done:
            action, carry = agent.take_action(obs, carry, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            returns[i] += reward
            eval_steps += 1

            if i == 0 and eval_steps % 2 == 0:
                frame = env.render()  				# (H, W, C), uint8
                h, w = frame.shape[:2]
                scale = 200 / min(h, w)
                if scale < 1.0:
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)	# 减少内存和存储占用
                frames.append(frame)
        
        total_steps += eval_steps
    
    video = np.array(frames)  # shape: (T, H, W, C)
    # 调整维度为 (N, T, C, H, W)
    video = torch.from_numpy(video).unsqueeze(0).permute(0, 1, 4, 2, 3).float() / 255.0
    return returns, total_steps/num_episodes, video

def save_checkpoint(logdir, agent, replay_buffer, step):
    checkpoint = {
        "step": step,
        "agent": agent.state_dict(),
        "agent.optim": agent.optim.state_dict(),
        "agent.scaler": agent.scaler.state_dict(),
        "replay_buffer": replay_buffer.state_dict()
    }
    torch.save(checkpoint, logdir + "/checkpoint.pt")

def load_checkpoint(logdir, agent, replay_buffer, device=None):
    path = logdir + "/checkpoint.pt"

    try:
        checkpoint = torch.load(path, map_location=device)
        step = checkpoint["step"]
        agent.load_state_dict(checkpoint["agent"])
        agent.optim.load_state_dict(checkpoint["agent.optim"])
        agent.scaler.load_state_dict(checkpoint["agent.scaler"])
        replay_buffer.load_state_dict(checkpoint["replay_buffer"])
    except:
        print(f"Unable to load checkpoint from {path}.")
        step = 0
    else:
        print(f"Successfully load checkpoint from {path}.")

    return step

if __name__ == "__main__":
    multiprocessing.set_start_method('fork')
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir")
    parser.add_argument("--task", default="BipedalWalker-v3", type=str, help="Gymnasium environment id."
                        "Examples: Pendulum-v1, Swimmer-v5, HalfCheetah-v5, BipedalWalker-v3, "
                        "Ant-v5, Humanoid-v5, ALE/Pong-v5, Grid, DMC-cartpole-balance, DMC-cheetah-run")
    parser.add_argument("--steps", default=1000000, type=int, help="Max number of steps that the agent interacts with the environment")
    parser.add_argument("--save", action="store_true", help="save checkpoint or not")
    parser.add_argument("--async_eval", action="store_true", help="use asynchronous evaluation or not. "
                        "When it's activated, evaluation will conduct on cpu asynchronously.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    args = parser.parse_args()

    tools.set_random_seed(args.seed)
    print(f"set random seed to {args.seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using device:", device)

    logdir = args.logdir or f"../runs/{args.task.replace('ALE/', '')}_RSAC_{str(time.time())[-5:]}"
    os.makedirs(logdir, exist_ok=True)
    print("logdir:", logdir)

    train_env = envs.make_env(args.task, seed=args.seed)
    eval_env = envs.make_env(args.task, seed=args.seed)
    obs_space = train_env.observation_space
    act_space = train_env.action_space
    print("task:", args.task)
    print("obs_space:", obs_space)
    print("act_space:", act_space)

    buffer_size = min(buffer_size, args.steps)
    replay_buffer = replay.ReplayBuffer(buffer_size, obs_space, act_space, seq_len, device, device)
    # 创建两个agent对象，一个用于训练，一个用于测试
    # 一方面，测试和训练可以同步进行；另一方面，测试和训练的batch_size不同，torch.compile可以分别优化
    train_agent = agent_RSAC.Agent(obs_space, hidden_dim, act_space, lr, grad_clip, alpha, tau, gamma, loss_scales, device)
    eval_agent = agent_RSAC.Agent(obs_space, hidden_dim, act_space, lr, grad_clip, alpha, tau, gamma, loss_scales, "cpu" if args.async_eval else device)
    train_agent = torch.compile(train_agent)
    eval_agent = torch.compile(eval_agent)
    step = load_checkpoint(logdir, train_agent, replay_buffer, device)

    # ========== print total params ========== 
    print("==========total params:{:,}==========".format(tools.get_params_num(train_agent)))
    if hasattr(train_agent, "encoder"):
        print("encoder: {:,}".format(tools.get_params_num(train_agent.encoder)))
    print("predictor: {:,}".format(tools.get_params_num(train_agent.predictor)))
    print("actor: {:,}".format(tools.get_params_num(train_agent.actor)))
    print("critic: {:,}".format(tools.get_params_num(train_agent.critic)))
    print("=====================================")
    
    print("begin to train.")
    train_policy(logdir, step, train_env, eval_env, train_agent, eval_agent, replay_buffer, 
              total_steps=args.steps, save=args.save, use_async_eval=args.async_eval)

    train_env.close()
    eval_env.close()