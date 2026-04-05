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

# todo: 考虑改为使用sns.lineplot来完成
def plot_comparison_curves(results_dict, interval, task, metric_disc, figsize=(6,4), xlim="auto", ylim="auto", legend=True):
    plt.figure(figsize=figsize)
    colors = ['#A51C36', '#682487', '#4485C7', '#DBB428', '#629C35']
    max_value = {}

    for idx, (algo_name, returns) in enumerate(results_dict.items()):
        n_runs, n_evals = returns.shape
        steps = np.arange(1, n_evals+1) * interval / 1000
        
        mean, std = returns.mean(axis=0), returns.std(axis=0)
        max_ind = np.argmax(mean)
        max_value[algo_name] = f"{mean[max_ind]:.2f}±{std[max_ind]:.2f}"

        smoothed = ema_on_axis1(returns, 0.8)
        mean, std = smoothed.mean(axis=0), smoothed.std(axis=0)
        plt.plot(steps, mean, color=colors[idx % len(colors)], linewidth=2.5, label=algo_name)
        plt.fill_between(steps, mean-std, mean+std, color=colors[idx % len(colors)], alpha=0.2)
    
    plt.xlabel('Env steps(k)', fontsize=13)
    plt.ylabel(metric_disc, fontsize=13)
    plt.title(task, fontsize=15, fontweight='bold', pad=20)
    if legend:
        plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')

    if xlim == "auto":
        plt.xlim((0, steps[-1]))
    else:
        plt.xlim(xlim)
    
    if ylim != "auto":
        plt.ylim(ylim)
    
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    # plt.subplots_adjust(left=0.2, right=0.95, bottom=0.1, top=0.9)
    plt.tight_layout()
    return max_value

def aggregate_and_plot(logdir, metric_key, metric_disc, figsize, xlim, ylim):
    # 从单次实验中提取metric_key对应数组
    def get_metric_array(log_name):
        steps = []
        metric_array = []
        with open(f"{logdir}/{log_name}/metrics.txt", "r") as log:
            for line in log:
                begin, end = line.find("{"), line.rfind("}")
                metrics = eval(line[begin:end+1])
                steps.append(metrics["step"])
                metric_array.append(metrics[metric_key])
        
        steps = np.array(steps)
        metric_array = np.array(metric_array)
        intervals = np.diff(steps)
        assert (intervals == intervals[0]).all(), "interval between evaluation points must be the same"
        return metric_array, intervals[0]
    
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
                
                metric_array, current_interval = get_metric_array(item)
                results[task][algorithm].append(metric_array)
                assert (not interval) or (interval == current_interval), "interval between evaluation points must be the same"
                interval = current_interval
    
    data, labels = [], []
    for task in results:
        for algorithm in results[task]:
            results[task][algorithm] = np.array(results[task][algorithm])
        # 保证每次画图时算法顺序相同，同一算法对应曲线颜色相同
        results[task] = dict(sorted(results[task].items()))
        max_value = plot_comparison_curves(results[task], interval, task, metric_disc, figsize, xlim, ylim)
        data.append(max_value)
        labels.append(task)
        plt.savefig(f'{logdir}/{task.replace("ALE/", "")}_{metric_key.replace("/","_")}.png', dpi=300, bbox_inches='tight')
    
    df = pd.DataFrame(data, index=labels)
    print(df)
    df.to_csv(f'{logdir}/max_value_{metric_disc}.csv')


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
    elif config["algorithm"] == "RSAC":
        cmd = ["python", "./RSAC/main_RSAC.py", 
            "--task", config["task"], 
            "--steps", str(config["steps"]),
            "--logdir", f"{config["logdir"]}/{log_name}", 
            "--seed", str(config["seed"])]
        if config["async_eval"]:
            cmd.append("--async_eval")
    elif config["algorithm"] == "VRM":
        cmd = ["python", "./VRM/run_experiment.py", 
           "--task", config["task"], 
           "--steps", str(config["steps"]),
           "--logdir", f"{config["logdir"]}/{log_name}", 
           "--seed", str(config["seed"])]
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

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="option")

# eval
parser_eval = subparsers.add_parser("eval", help="Execute algorithms for evaluation")
parser_eval.add_argument("--logdir", default="./runs", type=str, help="Select the directory of logs")
parser_eval.add_argument("--algorithm", nargs="+", type=str, choices=["SAC", "PPO", "TD3", "RSAC", "VRM", "Actor-Predictor"], 
                         help="Algorithms to evaluate")
parser_eval.add_argument("--task", nargs="+", type=str, default=["BipedalWalker-v3"], help="Gymnasium environment id of the task")
parser_eval.add_argument("--seed", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Random seed set")
parser_eval.add_argument("--steps", default=1000000, type=int, help="Max number of steps that the agent interacts with the environment")
parser_eval.add_argument("--num_workers", default=1, type=int, help="Number of parallel worker processes")

# plot
parser_plot = subparsers.add_parser("plot", help="Plot curves for comparision")
parser_plot.add_argument("--logdir", default="./runs", type=str, help="Select the directory of logs")
parser_plot.add_argument("--metric", default="eval_return", help="Which metric to plot")
parser_plot.add_argument("--desc", default="Average Return", help="Description of this metric")
parser_plot.add_argument("--figsize", nargs=2, default=[6, 4], type=int, help="Size of the figure")
parser_plot.add_argument("--xlim", nargs=2, default=[0, 0], type=int, help="X-axis limits [min, max]. Default (0, 0) for auto-scaling")
parser_plot.add_argument("--ylim", nargs=2, default=[0, 0], type=int, help="Y-axis limits [min, max]. Default (0, 0) for auto-scaling")
args = parser.parse_args()

for arg, value in args.__dict__.items():
    print(f"{arg}: {value}")

if args.option == "eval":
    evaluate_agent(args.task, args.algorithm, args.steps, args.logdir, args.seed, args.num_workers)
elif args.option == "plot":
    xlim = "auto" if args.xlim == [0, 0] else args.xlim
    ylim = "auto" if args.ylim == [0, 0] else args.ylim
    aggregate_and_plot(args.logdir, args.metric, args.desc, args.figsize, xlim, ylim)