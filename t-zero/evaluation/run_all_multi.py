import argparse
import json
import multiprocessing as mp
import os
import re
import subprocess
from typing import Optional

import numpy as np

from template_list import template_list


def task_clean(text):
    # Clean the text according to allowed characters for a task name
    return re.sub(r"[^\w\d\._]+", "_", text)


def get_task_name(dataset_name, subset_name, template_name):
    return task_clean(dataset_name + (f"_{subset_name}_" if subset_name is not None else "_") + template_name)


def read_log(log_path: str) -> Optional[float]:
    with open(log_path, "r", encoding="utf-8") as f:
        try:
            m = json.load(f)
        except json.JSONDecodeError:
            print("JSON decode error:", log_path)
            return None
        try:
            acc = float(m["evaluation"]["accuracy"])
        except (KeyError, ValueError):
            print("Parsing error:", log_path)
            return None
    return acc


def maybe_read_log(log_path: str) -> Optional[float]:
    if os.path.exists(log_path):
        return read_log(log_path)
    else:
        return None


q_gpu = mp.Queue()
args = None
cur_gpu = None


def run(job: tuple) -> dict:
    global q_gpu, args, cur_gpu
    model_name, output_dir, dataset_name, subset_name, template_name = job
    if cur_gpu is None:
        cur_gpu = q_gpu.get()
    new_env = os.environ.copy()
    new_env["CUDA_VISIBLE_DEVICES"] = str(cur_gpu)

    task_name = get_task_name(dataset_name, subset_name, template_name)
    log = maybe_read_log(os.path.join(output_dir, task_name + ".json"))
    while log is None:
        base_command = [
            'python', 'run_eval.py',
            '--model_name_or_path', model_name,
            '--output_dir', output_dir,
            '--use_slow_tokenizer',
            '--parallelize'
        ]
        if subset_name is None:
            command = base_command + [
                '--dataset_name', dataset_name,
                '--template_name', template_name
            ]
        else:
            command = base_command + [
                '--dataset_name', dataset_name,
                '--dataset_config_name', subset_name,
                '--template_name', template_name
            ]
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            env=new_env
        )
        log = maybe_read_log(os.path.join(output_dir, task_name + ".json"))
    return {
        "dataset_name": dataset_name,
        "subset_name": subset_name,
        "template_name": template_name,
        "accuracy": log,
    }


def collect_log_only(job: tuple) -> dict:
    _, output_dir, dataset_name, subset_name, template_name = job
    task_name = get_task_name(dataset_name, subset_name, template_name)
    log = maybe_read_log(os.path.join(output_dir, task_name + ".json"))
    assert log is not None
    return {
        "dataset_name": dataset_name,
        "subset_name": subset_name,
        "template_name": template_name,
        "accuracy": log,
    }


def main():
    global q_gpu, args

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--n_gpus", type=int, default=1)
    args = parser.parse_args()

    jobs = []
    for dataset_name, subset_name in template_list:
        for template_name in template_list[(dataset_name, subset_name)]:
            jobs.append((args.model_name, args.output_dir, dataset_name, subset_name, template_name))

    unfinished_jobs = []
    for model_name, output_dir, dataset_name, subset_name, template_name in jobs:
        task_name = get_task_name(dataset_name, subset_name, template_name)
        log = maybe_read_log(os.path.join(output_dir, task_name + ".json"))
        if log is None:
            unfinished_jobs.append((model_name, output_dir, dataset_name, subset_name, template_name))
    print(f"Detected {len(unfinished_jobs)} unfinished jobs")

    for i in range(args.n_gpus):
        q_gpu.put(i)
    with mp.Pool(args.n_gpus) as pool:
        pool.map(run, unfinished_jobs)
    with mp.Pool(args.n_gpus) as pool:
        all_logs = pool.map(collect_log_only, jobs)

    result = {}
    for log in all_logs:
        key = (log["dataset_name"], log["subset_name"], log["template_name"])
        assert key not in result
        result[key] = log["accuracy"] * 100.0
    all_acc_s = []
    for dataset_name, subset_name in template_list:
        dataset_acc_s = []
        for template_name in template_list[(dataset_name, subset_name)]:
            dataset_acc_s.append(result[(dataset_name, subset_name, template_name)])
        dataset_acc = np.mean(dataset_acc_s)
        print(dataset_name, subset_name, dataset_acc)
        all_acc_s.append(dataset_acc)
    print('Overall Acc:', np.mean(all_acc_s))
    print(",".join(list(map(str, all_acc_s))))


if __name__ == '__main__':
    main()
