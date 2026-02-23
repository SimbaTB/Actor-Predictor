# Actor-Predictor

Implementation of Actor Predictor: Improving Policy Learning And Feature Extraction Through World Model, which combines LSTM-based world model with Soft Actor-Critic (SAC) by using the prediction of world model and the real observation from environment jointly as the input of value network and policy network, achieving outstanding performance and robustness.

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
### Actor-Predictor
To run training using the Actor-Predictor algorithm, you should execute `main.py` using following commands.
```
python main.py [--logdir LOGDIR] [--task TASK] [--steps STEPS] [--save] [--seed SEED]
```
- `--logdir`: Where to save logs and metrics
- `--task`: Gymnasium environment id. Examples: Pendulum-v1, Hopper-v5, BipedalWalker-v3, BipedalWalkerHardcore-v3, Walker2d-v5, Ant-v5, Humanoid-v5, ALE/Pong-v5, Grid, DMC-cartpole-balance, DMC-cheetah-run
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

### View Results
Both `main.py` and `baseline.py` use Tensorboard to log. To view the results, start tensorboard with the `--logdir` pointing to the log directory. Example:
```
tensorboard --logdir ./runs/BipedalWalker-v3_66157
```

## Evaluate And Plot Learning Curves Automatically
To evaluate and compare algorithms, you can use `evaluate.py`, which is able to evaluate specific algorithms on specific tasks with specific random seeds randomly and plot the learning curves.

For example, to compare the performance of Actor-Predictor, SAC, TD3, and PPO on BipedalWalker-v3, Hopper-v5 with random seeds 0, 1, 2, 3, 4:
```
python evaluate.py --option eval --logdir ./runs --algorithm SAC PPO TD3 Actor-Predictor --task BipedalWalker-v3 Hopper-v5 --steps 1000000 --num_workers 6 --seed 0 1 2 3 4
```

And when it's finished, all logs and metrics will be saved to `./runs`. Then you can plot learning curves with following command. The learning curves will be saved to logdir.
```
python evaluate.py --option plot --logdir ./runs 
```

Or you can also merge the two steps above together:
```
python evaluate.py --option all --logdir ./runs --algorithm SAC PPO TD3 Actor-Predictor --task BipedalWalker-v3 Hopper-v5 --steps 1000000 --num_workers 6 --seed 0 1 2 3 4
```
