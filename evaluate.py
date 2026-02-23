import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor
import os
import time
import random

def ema_on_axis1(x, alpha):
    assert len(x.shape) == 2    # x.shape: (B, T)
    x = np.asarray(x)
    y = np.empty_like(x)
    y[:,0] = x[:,0]
    for i in range(1, x.shape[1]):
        y[:,i] = alpha * x[:,i] + (1 - alpha) * y[:,i-1]
    return y

def plot_comparison_curves(results_dict, interval, task, legend=True):
    plt.figure(figsize=(6, 4))
    colors = ['#A51C36', '#682487', '#4485C7', '#DBB428', '#629C35']
    best_return = {}

    for idx, (algo_name, returns) in enumerate(results_dict.items()):
        n_runs, n_evals = returns.shape
        steps = np.arange(1, n_evals+1) * interval / 1000
        
        mean, std = returns.mean(axis=0), returns.std(axis=0)
        max_ind = np.argmax(mean)
        best_return[algo_name] = f"{mean[max_ind]:.2f}±{std[max_ind]:.2f}"

        smoothed = ema_on_axis1(returns, 0.8)
        mean, std = smoothed.mean(axis=0), smoothed.std(axis=0)
        plt.plot(steps, mean, color=colors[idx % len(colors)], linewidth=2.5, label=algo_name)
        plt.fill_between(steps, mean-std, mean+std, color=colors[idx % len(colors)], alpha=0.2)
    
    plt.xlabel('Env steps(k)', fontsize=13)
    plt.ylabel('Average Return', fontsize=13)
    plt.title(task, fontsize=15, fontweight='bold', pad=20)
    if legend:
        plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim((0, steps[-1]))
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    
    plt.tight_layout()
    return best_return

def run_benchmark(config):
    log_name = f"{config["task"].replace("ALE/", "")}_{config["algorithm"]}_seed{config["seed"]}"

    if config["algorithm"] == "Actor-Predictor":
        cmd = ["python", "main.py", 
            "--task", config["task"], 
            "--steps", str(config["steps"]),
            "--logdir", f"{config["logdir"]}/{log_name}", 
            "--seed", str(config["seed"])]
        if config["async_eval"]:
            cmd.append("--async_eval")
    else:
        cmd = ["python", "baseline.py", 
           "--algorithm", config["algorithm"], 
           "--task", config["task"], 
           "--steps", str(config["steps"]),
           "--logdir", f"{config["logdir"]}/{log_name}", 
           "--seed", str(config["seed"])]
    # 随机等待一段时间再开始，防止不同进程扎堆使用同一资源
    time.sleep(random.randint(0,10))
    print("execute: ", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        return f"{config["algorithm"]} on {config["task"]}: success"
    else:
        return f"{config["algorithm"]} on {config["task"]}: failed - {result.stderr}"
        
def evaluate_agent(tasks, algorithms, steps, logdir, seeds=range(5), max_workers=1):
    experiments = []
    for seed in seeds:
        for task in tasks:
            for algorithm in algorithms:
                experiments.append({"algorithm":algorithm, "task":task, "steps":steps, "logdir":logdir, "seed":seed, 
                                    "async_eval":False})

    # 并行运行（多实验同时跑）
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_benchmark, experiments))
        
    for r in results:
        print(r)

def aggregate_and_plot(logdir):
    # 从单次实验中提取eval_return数组
    def get_eval_return(log_name):
        steps = []
        eval_returns = []
        with open(f"{logdir}/{log_name}/metrics.txt", "r") as log:
            for line in log:
                begin, end = line.find("{"), line.rfind("}")
                metrics = eval(line[begin:end+1])
                steps.append(metrics["step"])
                eval_returns.append(metrics["eval_return"])
        
        steps = np.array(steps)
        eval_returns = np.array(eval_returns)
        intervals = np.diff(steps)
        assert (intervals == intervals[0]).all(), "interval between evaluation points must be the same"
        return eval_returns, intervals[0]
    
    # 期望logdir下每次实验格式：{task}_{algorithm}_seed{seed}
    results = {}
    interval = None
    for item in os.listdir(logdir): 
        if os.path.isdir(os.path.join(logdir, item)):
            try:
                task, algorithm, seed = item.split("_")
            except Exception as e:
                print(f"Exception occurred when procecing {item}:", e)
            else:
                if task not in results:
                    results[task] = {}
                
                if algorithm not in results[task]:
                    results[task][algorithm] = []
                
                eval_returns, current_interval = get_eval_return(item)
                results[task][algorithm].append(eval_returns)
                assert (not interval) or (interval == current_interval), "interval between evaluation points must be the same"
                interval = current_interval
    
    data, labels = [], []
    for task in results:
        for algorithm in results[task]:
            results[task][algorithm] = np.array(results[task][algorithm])
        # 保证每次画图时算法顺序相同，同一算法对应曲线颜色相同
        results[task] = dict(sorted(results[task].items()))
        best_return = plot_comparison_curves(results[task], interval, task)
        data.append(best_return)
        labels.append(task)
        plt.savefig(f'{logdir}/{task.replace("ALE/", "")}_results.png', dpi=300, bbox_inches='tight')
    
    df = pd.DataFrame(data, index=labels)
    print(df)
    df.to_csv(f'{logdir}/best_returns.csv')

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", default="./runs", type=str, help="Select the directory of logs")
parser.add_argument("--algorithm", nargs="+", type=str, default=["SAC", "PPO", "TD3", "Actor-Predictor"], 
                    help="Algorithms to evaluate")
parser.add_argument("--option", type=str, choices=["eval", "plot", "all"], help="Action to do")
parser.add_argument("--task", nargs="+", type=str, default=["BipedalWalker-v3"], help="Gymnasium environment id of the task")
parser.add_argument("--steps", default=1000000, type=int, help="Max number of steps that the agent interacts with the environment")
parser.add_argument("--num_workers", default=1, type=int, help="Number of parallel worker processes")
parser.add_argument("--seed", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Random seed set")
args = parser.parse_args()

if args.option in ["eval", "all"] and not args.task:
    raise argparse.ArgumentError(None, "Please specify the task")

if args.option in ["plot", "all"] and not args.logdir:
    raise argparse.ArgumentError(None, "Please specify logdir")

for arg, value in args.__dict__.items():
    print(f"{arg}: {value}")

if args.option == "eval":
    evaluate_agent(args.task, args.algorithm, args.steps, args.logdir, args.seed, args.num_workers)
elif args.option == "plot":
    aggregate_and_plot(args.logdir)
elif args.option == "all":
    evaluate_agent(args.task, args.algorithm, args.steps, args.logdir, args.seed, args.num_workers)
    aggregate_and_plot(args.logdir)