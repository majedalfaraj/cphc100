import argparse
import json
import random
import os, subprocess
from csv import DictWriter
import multiprocessing
import itertools
import sys

def add_main_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--model_name', type=str, default='cnn',
                       choices=['mlp', 'cnn'], # TODO: add your models names here
                       help='Model to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for training')
    parser.add_argument('--num_epochs', type=int, default=1,
                       help='Number of epochs to train')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay for regularization')
    parser.add_argument('--config_path', type=str, default='grid_search.json',
                        help='Path to grid search JSON configuration file')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of worker processes to run in parallel')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory to save experiment logs and results')
    parser.add_argument('--grid_search_results_path', type=str, default='grid_results.csv',
                        help='Path to final CSV file collecting all experiment results')
    parser.add_argument('--max_steps_per_epoch', type=int, default=10000,
                    help='Maximum number of steps (batches) per epoch for faster exploration')

    return parser

def get_experiment_list(config: dict) -> list[dict]:
    '''
    Parses an experiment config, and creates jobs. For flags that are expected to be a single item, 
    but the config contains a list, this will return one job for each item in the list.
    
    Args:
        config: experiment configuration dictionary from grid_search.json
        
    Example config:
    {
        "learning_rate": [0.001, 0.01],
        "batch_size": [64, 128],
        "regularization_lambda": [0, 0.1]
    }
    
    Returns:
        jobs: a list of dicts, each of which encapsulates one job.
        Example: [
            {"learning_rate": 0.001, "batch_size": 64, "regularization_lambda": 0},
            {"learning_rate": 0.001, "batch_size": 64, "regularization_lambda": 0.1},
            {"learning_rate": 0.001, "batch_size": 128, "regularization_lambda": 0},
            ...
        ]
    '''
    jobs = [{}]

    def combination_tree(config):
        if len(config) == 0:
            return [{}]
        
        config_copy = config.copy()
        key = list(config_copy.keys())[0]
        values = config_copy.pop(key)
        old_combos = combination_tree(config_copy)
        new_combos = []
        for v in values:
            for d in old_combos:
                d_copy = d.copy()
                d_copy[key] = v
                new_combos.append(d_copy)
        return new_combos

    jobs = combination_tree(config)

    return jobs

def worker(args: argparse.Namespace, job_queue: multiprocessing.Queue, done_queue: multiprocessing.Queue, problem: str):
    '''
    Worker thread for each worker. Consumes all jobs and pushes results to done_queue.
    :args - command line args
    :job_queue - queue of available jobs.
    :done_queue - queue where to push results.
    '''
    while not job_queue.empty():
        params = job_queue.get()
        if params is None:
            return
        done_queue.put(
            launch_experiment(args, params, problem))

def launch_experiment(args: argparse.Namespace, experiment_config: dict, problem) -> dict:
    '''
    Launch an experiment and direct logs and results to a unique filepath.
    
    Args:
        args: command line arguments
        experiment_config: flags to use for this model run. Will be fed into main.py
        
    Returns:
        dict: flags for this experiment as well as result metrics
        
    Example return:
    {
        "learning_rate": 0.001,
        "batch_size": 64, 
        "regularization_lambda": 0.1,
        "train_auc": 0.65,
        "val_auc": 0.62
    }
    '''

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    # TODO: Launch the experiment
    exp_name = "_".join([f"{k}-{v}" for k, v in experiment_config.items()])
    exp_dir = os.path.join(args.log_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    results_path = os.path.join(exp_dir, "results.json")
    command = [
        sys.executable,
        f"{problem}_main.py",
        f"--results_path={results_path}",  # Add results_path explicitly
    ]

    for k, v in experiment_config.items():
        command.append(f"--{k}={v}")

    try:
        # ✅ Run the experiment and capture its stdout
        result = subprocess.run(command, capture_output=True, text=True)

        metrics = {}
        stdout_lines = result.stdout.strip().splitlines()

        # Look for the last valid JSON line in stdout
        for line in reversed(stdout_lines):
            try:
                metrics = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

        if not metrics:
            print(f"[WARN] No valid JSON metrics found for {experiment_config}")

        return {**experiment_config, **metrics}
    
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Experiment {exp_name} failed: {e}")
        return {**experiment_config, "train_auc": None, "val_auc": None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CPH 100A Project 2 - PathMNIST Classification')
    parser = add_main_args(parser)
    args = parser.parse_args()
    return args

def main(args: argparse.Namespace, problem: str) -> list[dict]:
    print(args)
    config = json.load(open(args.config_path, "r"))
    print("Starting grid search with the following config:")
    print(json.dumps(config, indent=2))

    # TODO: From config, generate a list of experiments to run
    experiments = get_experiment_list(config)
    random.shuffle(experiments)

    job_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()

    for exper in experiments:
        job_queue.put(exper)

    print("Launching dispatcher with {} experiments and {} workers".format(len(experiments), args.num_workers))

    # TODO: Define worker fn to launch an experiment as a separate process.
    for _ in range(args.num_workers):
        multiprocessing.Process(target=worker, args=(args, job_queue, done_queue, problem)).start()

    # Accumualte results into a list of dicts
    grid_search_results = []
    for _ in range(len(experiments)):
        grid_search_results.append(done_queue.get())

    keys = grid_search_results[0].keys()

    print("Saving results to {}".format(args.grid_search_results_path))

    writer = DictWriter(open(args.grid_search_results_path, 'w'), keys)
    writer.writeheader()
    writer.writerows(grid_search_results)

    print("Done")
    return grid_search_results

if __name__ == '__main__':
    __spec__ = None
    args = parse_args()
    main(args, "classification")