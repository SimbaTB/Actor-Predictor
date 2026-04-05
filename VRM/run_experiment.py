import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from vrdm import VRM, VRDM
import time, os, argparse, warnings
import scipy.io as sio
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs

def test_performance(agent_test, env_test, action_filter, times=10):
    agent_test.eval()
    EpiTestRet = 0
    with torch.no_grad():
        for _ in range(times):
            s0, _ = env_test.reset()
            s0 = s0.astype(np.float32)
            r0 = np.array([0.], dtype=np.float32)
            x0 = np.concatenate([s0, r0])
            a = agent_test.init_episode(x0).reshape(-1)

            for t in range(max_steps):
                if np.any(np.isnan(a)):
                    raise ValueError
                sp, r, terminated, truncated, _ = env_test.step(action_filter(a))
                a = agent_test.select(sp, r, action_return='normal')
                EpiTestRet += r
                if terminated or truncated:
                    break
    return EpiTestRet / times

parser = argparse.ArgumentParser()
parser.add_argument("--logdir")
parser.add_argument("--task", default="Pendulum-v1", type=str, help="Gymnasium environment id.")
parser.add_argument("--steps", default=100000, type=int, help="Max number of steps that the agent interacts with the environment")
parser.add_argument("--seed", default=0, type=int, help="Random seed")
parser.add_argument("--render", action="store_true")
args = parser.parse_args()

logdir = args.logdir or f"../runs/{args.task.replace('ALE/', '')}_VRM_{str(time.time())[-5:]}"
print("logdir:", logdir)
if os.path.exists(logdir):
    warnings.warn('{} exists (possibly so do data).'.format(logdir))
else:
    os.makedirs(logdir)

print("task:", args.task)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

seed = args.seed # random seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

beta_h = 'auto_1.0'
optimizer_st = 'adam'
# minibatch_size = 4
# seq_len = 64
minibatch_size = 16
seq_len = 16
reward_scale = 1.0
lr_vrm = 8e-4
gamma = 0.99
max_all_steps = args.steps  # total steps to learn
step_perf_eval = 4000  # how many steps to do evaluation
rendering = args.render


env = envs.make_env(args.task, seed=seed)
env_test = envs.make_env(args.task, seed=seed)
action_filter = lambda a: a.reshape([-1])
max_steps = 1000
est_min_steps = 5

rnn_type = 'mtlstm'
d_layers = [128, ]
z_layers = [64, ]
x_phi_layers = [128]
decode_layers = [128, 128]

value_layers = [128, 128]
policy_layers = [128, 128]

step_start_rl = 1000
step_start_st = 1000
step_end_st = np.inf
fim_train_times = 5000

train_step_rl = 1  # how many times of RL training after step_start_rl
train_step_st = 5

train_freq_rl = 1. / train_step_rl
train_freq_st = 1. / train_step_st

max_episodes = int(max_all_steps / est_min_steps) + 1  # for replay buffer

fim = VRM(input_size=env.observation_space.shape[0] + 1,
          action_size=env.action_space.shape[0],
          rnn_type=rnn_type,
          d_layers=d_layers,
          z_layers=z_layers,
          decode_layers=decode_layers,
          x_phi_layers=x_phi_layers,
          optimizer=optimizer_st,
          lr_st=lr_vrm).to(device)

klm = VRM(input_size=env.observation_space.shape[0] + 1,
          action_size=env.action_space.shape[0],
          rnn_type=rnn_type,
          d_layers=d_layers,
          z_layers=z_layers,
          decode_layers=decode_layers,
          x_phi_layers=x_phi_layers,
          optimizer=optimizer_st,
          lr_st=lr_vrm).to(device)

agent = VRDM(fim, klm, gamma=gamma,
             beta_h=beta_h,
             value_layers=value_layers,
             policy_layers=policy_layers).to(device)

agent_test = VRDM(fim, klm, gamma=gamma,
                  beta_h=beta_h,
                  value_layers=value_layers,
                  policy_layers=policy_layers).to(device)

SP_real = np.zeros([max_episodes, max_steps, env.observation_space.shape[0]], dtype=np.float32)  # observation (t+1)
A_real = np.zeros([max_episodes, max_steps, env.action_space.shape[0]], dtype=np.float32)  # action
R_real = np.zeros([max_episodes, max_steps], dtype=np.float32)  # reward
D_real = np.zeros([max_episodes, max_steps], dtype=np.float32)  # done
V_real = np.zeros([max_episodes, max_steps], dtype=np.float32)  # mask, indicating whether a step is valid. value: 1 (compute gradient at this step) or 0 (stop gradient at this step)

performance_wrt_step = []
global_steps = []

e_real = 0
global_step = 0
last_step, last_time = 0, time.time()
loss_sts = []
writer = SummaryWriter(log_dir=logdir)

#  Run
while global_step < max_all_steps:

    s0, _ = env.reset()
    s0 = s0.astype(np.float32)
    r0 = np.array([0.], dtype=np.float32)
    x0 = np.concatenate([s0, r0])
    a = agent.init_episode(x0).reshape(-1)

    for t in range(max_steps):

        if global_step == max_all_steps:
            break

        if np.any(np.isnan(a)):
            raise ValueError

        sp, r, terminated, truncated, _ = env.step(action_filter(a))
        if rendering:
            env.render()

        A_real[e_real, t, :] = a
        SP_real[e_real, t, :] = sp.reshape([-1])
        R_real[e_real, t] = r
        D_real[e_real, t] = 1 if terminated else 0
        V_real[e_real, t] = 1

        a = agent.select(sp, r)

        global_step += 1

        if global_step == step_start_st + 1:
            print("Start training the first-impression model!")
            _, _, loss_st = agent.learn_st(True, False,
                                           SP_real[0:e_real], A_real[0:e_real], R_real[0:e_real],
                                           D_real[0:e_real], V_real[0:e_real],
                                           times=fim_train_times, minibatch_size=minibatch_size)
            loss_sts.append(loss_st)
            print("Finish training the first-impression model!")
            print("Start training the keep-learning model!")

        if global_step > step_start_st and global_step <= step_end_st and np.random.rand() < train_freq_st:
            _, _, loss_st = agent.learn_st(False, True,
                                           SP_real[0:e_real], A_real[0:e_real], R_real[0:e_real],
                                           D_real[0:e_real], V_real[0:e_real],
                                           times=max(1, int(train_freq_st)), minibatch_size=minibatch_size)
            loss_sts.append(loss_st)

        if global_step > step_start_rl and np.random.rand() < train_freq_rl:
            if global_step == step_start_rl + 1:
                print("Start training the RL controller!")
            agent.learn_rl_sac(SP_real[0:e_real], A_real[0:e_real], R_real[0:e_real],
                               D_real[0:e_real], V_real[0:e_real],
                               times=max(1, int(train_freq_rl)), minibatch_size=minibatch_size,
                               reward_scale=reward_scale, seq_len=seq_len)

        if global_step % step_perf_eval == 0:
            agent_test.load_state_dict(agent.state_dict())  # update agent_test
            EpiTestRet = test_performance(agent_test, env_test, action_filter, times=5)
            performance_wrt_step.append(EpiTestRet)
            global_steps.append(global_step)

            current_time = time.time()
            fps = (global_step - last_step) / (current_time - last_time)
            last_step, last_time = global_step, current_time
            metrics = {"step": global_step, "episode": e_real, "episode_len": t, "eval_return": float(EpiTestRet), "fps": fps}
            for name, value in metrics.items():
                writer.add_scalar(name, value, global_step)
            print(metrics)

            with open(logdir + "/metrics.txt", "a") as file:
                file.write(str(metrics) + "\n")

        if terminated or truncated:
            break

    print(args.task + " - episode {} : steps {}, mean reward {}".format(e_real, t, np.mean(R_real[e_real])))
    e_real += 1

writer.close()
performance_wrt_step_array = np.reshape(performance_wrt_step, [-1]).astype(np.float64)
global_steps_array = np.reshape(global_steps, [-1]).astype(np.float64)
loss_sts = np.reshape(loss_sts, [-1]).astype(np.float64)

data = {"loss_sts": loss_sts,
        "max_episodes": max_episodes,
        "step_start_rl": step_start_rl,
        "step_start_st": step_start_st,
        "step_end_st": step_end_st,
        "rnn_type": rnn_type,
        "optimizer": optimizer_st,
        "reward_scale": reward_scale,
        "beta_h": beta_h,
        "minibatch_size": minibatch_size,
        "train_step_rl": train_step_rl,
        "train_step_st": train_step_st,
        "R": np.sum(R_real, axis=-1).astype(np.float64),
        "steps": np.sum(V_real, axis=-1).astype(np.float64),
        "performance_wrt_step": performance_wrt_step_array,
        "global_steps": global_steps_array}

sio.savemat(logdir + "/" + args.task + "_vrm.mat", data)
torch.save(agent, logdir + "/" + args.task + "_vrm.model")