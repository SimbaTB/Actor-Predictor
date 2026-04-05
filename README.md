# Actor-Predictor

Implementation of `Improving Policy Learning and Robustness in Partially Observable Reinforcement Learning through World Model`, which combines LSTM-based world model with Soft Actor-Critic (SAC) by using the prediction of world model and the real observation from environment jointly as the input of value network and policy network, achieving outstanding performance and robustness.

## Install Dependencies
This code has been tested on Python 3.12. Run following commands to install dependencies.

```
conda create -n env1 python=3.12
conda activate env1
sudo apt-get update
sudo apt-get install -y swig libegl1-mesa libegl1-mesa-dev libosmesa6-dev
pip install -r requirements.txt
```

## Run Training Manually

### Available Environment IDs
Available environment IDs include but are not limited to: 
- **Classic control** (vector observation): `Pendulum-v1`, `CartPoleContinuous-v1`, `MountainCarContinuous-v0`
- **Box2D** (vector observation): `BipedalWalker-v3`, `BipedalWalkerHardcore-v3`
- **MuJoCo** (vector observation): `Hopper-v5`, `Walker2d-v5`, `Ant-v5`, `Humanoid-v5`
- **dm_control** (image observation): `DMC-cartpole-balance`, `DMC-cheetah-run`
- **Atari** (image observation): `ALE/Pong-v5`
- **Others**: `Grid`(image observation)

To introduce observation noise $\varepsilon\sim \mathcal N(0,0.1^2)$, add `-N` to the ID of environments with vector-based observation, example: `CartPoleContinuous-v1-N`

To introduce partial observability (positions only), add `-P`. Available choices: `Pendulum-v1-P`, `MountainCarContinuous-v0-P`, `CartPoleContinuous-v1-P`

### Actor-Predictor
To run training using the Actor-Predictor algorithm, you should execute `main.py` using following commands.
```
python main.py [--logdir LOGDIR] [--task TASK] [--steps STEPS] [--save] [--seed SEED]
```
- `--logdir`: Where to save logs and metrics
- `--task`: Gymnasium environment id. 
- `--steps`: Total timesteps that the agent interacts with the environment
- `--save`:  Whether to save checkpoint or not. When checkpoint is saved, running the same command with the same `logdir` can resume training from checkpoint.
- `--seed`: Set random seed

For example, to train an agent using the Actor-Predictor algorithm on the BipedalWalker-v3 task for 1000000 timesteps with random seed 0, run following command:
```
python main.py --task BipedalWalker-v3 --steps 1000000 --save --seed 0 
```

### Baselines
To run training using the baseline algorithms, you should execute `baseline.py` using following commands.
```
python baseline.py [--algorithm {PPO,SAC,TD3,RecurrentPPO}] [--logdir LOGDIR] [--task TASK] [--steps STEPS] [--seed SEED]
```
- `--algorithm`: Algorithm name (PPO,SAC,TD3,RecurrentPPO)
- `--logdir`: The same as `main.py`
- `--task`: The same as `main.py`
- `--steps`: The same as `main.py`
- `--seed`: The same as `main.py`

For example, to train an agent using the SAC algorithm on the BipedalWalker-v3 task for 1000000 timesteps with random seed 0, run following command:
```
python baseline.py --algorithm SAC --task BipedalWalker-v3 --steps 1000000 --seed 0 
```

### Recurrent SAC (RSAC)
To run the RSAC algorithm, please execute following command. Parameters are just the same as `main.py`
```
python ./RSAC/main_RSAC.py [--logdir LOGDIR] [--task TASK] [--steps STEPS] [--save] [--seed SEED]
```

### Variational Recurrent Models (VRM)
To run the VRM algorithm, please execute the following command. Note that our implementation of VRM is a modified version of the original code (https://github.com/oist-cnru/Variational-Recurrent-Models), adapted for GPU compatibility.
```
python ./VRM/run_experiment.py [--logdir LOGDIR] [--task TASK] [--steps STEPS] [--seed SEED]
```

### View Results
All the programs above use Tensorboard to log. To view the results, start tensorboard with the `--logdir` pointing to the log directory. Example:
```
tensorboard --logdir ./runs/BipedalWalker-v3_66157
```

In addition, all the programs above also save metrics to `LOGDIR/metrics.txt`.

## Evaluate And Plot Learning Curves Automatically
To evaluate and compare algorithms, you can use `evaluate.py`, which is able to evaluate specific algorithms on specific tasks with specific random seeds and plot the learning curves.

This tool supports two operation modes: **Algorithm Evaluation** and **Result Plotting**.

### Algorithm Evaluation
To automatically run the above algorithms in batches across multiple environments, multiple algorithms, and multiple seeds, you can use this tool.
```
python evaluate.py eval [--logdir LOGDIR] [--algorithm ALGORITHM ...] [--task TASK ...] [--seed SEED ...] [--steps STEPS] [--num_workers NUM_WORKERS]
```
- `--algorithm`: List of algorithm(s) to evaluate. Available options: SAC, PPO, TD3, RSAC, VRM, Actor-Predictor
- `--task`: List of gymnasium environment ID(s).
- `--logdir` (Optional, default="./runs"): Directory to save logs and metrics
- `--seed` (Optional, default=[0,1,2,3,4]): List of random seed(s).
- `--steps` (Optional, default=1000000): Maximum number of steps for agent-environment interaction
- `--num_workers` (Optional, default=1): Number of parallel worker processes

For example, to compare the performance of Actor-Predictor, SAC, TD3, and PPO on BipedalWalker-v3, Hopper-v5 with random seeds 0, 1, 2, 3, 4:
```
python evaluate.py eval --logdir ./runs --algorithm SAC PPO TD3 Actor-Predictor --task BipedalWalker-v3 Hopper-v5 --steps 1000000 --num_workers 6 --seed 0 1 2 3 4
```
And when it's finished, all logs and metrics will be saved to `./runs`. 

### Result Plotting
In this mode, `evaluate.py` will read `LOGDIR/metrics.txt` and plot the `METRIC` you specified.
```
python evaluate.py plot [--logdir LOGDIR] [--metric METRIC] [--desc DESC] [--figsize WIDTH HEIGHT] [--xlim MIN MAX] [--ylim MIN MAX]
```
- `--logdir` (Optional, default="./runs"): Directory containing log data
- `--metric` (Optional, default="eval_return"): Name of the metric to plot
- `--desc` (Optional, default="Average Return"): Description of this metric (shown in legend/axes)
- `--figsize` (Optional, default=[6,4]): Figure size [width, height]
- `--xlim` (Optional, default=[0,0]): X-axis limits [min, max]. Set to [0,0] for auto-scaling
- `--ylim` (Optional, default=[0,0]): Y-axis limits [min, max]. Set to [0,0] for auto-scaling

For example, you can plot learning curves with the following command. The learning curves will be saved to `./runs`.
```
python evaluate.py plot --logdir ./runs 
```

## Third-Party Code

`VRM/` directory contains code adapted from:
- Repository: [oist-cnru/Variational-Recurrent-Models](https://github.com/oist-cnru/Variational-Recurrent-Models)
- License: [LICENSE-VRM](variational_recurrent_models/LICENSE) (included in directory)
- Modifications: 
    - Added CUDA support and automatic device detection
    - Modified command-line interface and argument parsing