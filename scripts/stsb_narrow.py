import json
import os

import matplotlib.pyplot as plt
import numpy as np

from dlt import configs, metrics, plot_utils
from dlt.finetune import finetune


def common_config():
    task_config = configs.TaskConfig()
    task_config.finetune_task_name = configs.GlueTaskName.STSB
    task_config.num_train_samples = 16
    task_config.train_batch_size = 16
    task_config.max_seq_length = 128
    task_config.lora_init_scale = 1e-3
    task_config.num_train_steps = 200
    task_config.log_eval_steps = 20
    task_config.decay_ratio = 1.0
    task_config.lora_depth = 3
    task_config.lora_gamma = 1e-2
    task_config.lora_adapt_type = configs.LoraAdaptType.ONLY_QUERY_VALUE
    task_config.save_dir = f"checkpoints/stsb_fewshot_16_narrow_vs_wide"
    return task_config


def run_experiments(rank, seeds, compress, random, learning_rate):
    task_config = common_config()
    task_config.lora_rank = rank
    task_config.lora_compress = compress
    task_config.lora_random_factors = random
    task_config.learning_rate = learning_rate
    finetune(task_config, seeds)


def main():
    seeds = [0]
    run_experiments(
        rank=None, seeds=seeds, compress=False, random=False, learning_rate=1e-4
    )
    run_experiments(
        rank=8, seeds=seeds, compress=False, random=False, learning_rate=1e-4
    )
    run_experiments(rank=8, seeds=seeds, compress=True, random=True, learning_rate=1e-2)


def read(experiment_path):
    with open(os.path.join(experiment_path, "results.json")) as f:
        results = json.load(f)
    tags = results.keys()
    result_dict = {}
    for tag in tags:
        step_vals, value_vals = list(
            zip(*[(pair["step"], pair["value"]) for pair in results[tag]])
        )
        result_dict[tag] = (np.array(step_vals), np.array(value_vals))
    return result_dict


def get_results():
    experiment_dir = "checkpoints/stsb_fewshot_16_narrow_vs_wide"
    runs = os.listdir(experiment_dir)
    random_run = [x for x in runs if "rank=8_samples=16" in x][0]
    compress_run = [x for x in runs if "compress_samples" in x][0]
    original_run = [x for x in runs if "rank=8" not in x][0]
    random_results = read(os.path.join(experiment_dir, random_run))
    compress_results = read(os.path.join(experiment_dir, compress_run))
    original_results = read(os.path.join(experiment_dir, original_run))

    train_step_vals, random_train_loss_vals = random_results["train_loss"]
    _, compress_train_loss_vals = compress_results["train_loss"]
    _, original_train_loss_vals = original_results["train_loss"]

    eval_step_vals, random_eval_vals = random_results[metrics.GLUE_METRIC_DICT["stsb"]]
    _, compress_eval_vals = compress_results[metrics.GLUE_METRIC_DICT["stsb"]]
    _, original_eval_vals = original_results[metrics.GLUE_METRIC_DICT["stsb"]]

    return (
        train_step_vals,
        random_train_loss_vals,
        compress_train_loss_vals,
        original_train_loss_vals,
        eval_step_vals,
        random_eval_vals,
        compress_eval_vals,
        original_eval_vals,
    )


def plot_results():
    (
        train_step_vals,
        random_train_loss_vals,
        compress_train_loss_vals,
        original_train_loss_vals,
        eval_step_vals,
        random_eval_vals,
        compress_eval_vals,
        original_eval_vals,
    ) = get_results()
    fig, ax = plt.subplots(ncols=2, figsize=(9, 3))

    random_color = "deeppink"

    train_smooth_fn = lambda x: plot_utils.smooth(x, 0.99)
    ax[0].plot(
        train_step_vals,
        train_smooth_fn(original_train_loss_vals),
        "--",
        color="green",
        linewidth=4,
        label="Original",
    )
    ax[0].plot(
        train_step_vals,
        train_smooth_fn(random_train_loss_vals),
        color=random_color,
        label="Random",
    )
    ax[0].plot(
        train_step_vals,
        train_smooth_fn(compress_train_loss_vals),
        color="b",
        label="Compressed",
    )
    ax[0].set_xlabel("Iteration", fontsize=14)
    ax[0].set_ylabel("Train Loss", fontsize=14)
    ax[0].legend(fontsize=12)

    eval_smooth_fn = lambda x: plot_utils.smooth(x, 0.95)
    ax[1].plot(
        eval_step_vals,
        eval_smooth_fn(original_eval_vals),
        "--",
        color="green",
        linewidth=4,
        label="Original",
    )
    ax[1].plot(
        eval_step_vals,
        eval_smooth_fn(random_eval_vals),
        color=random_color,
        label="Random",
    )
    ax[1].plot(
        eval_step_vals,
        eval_smooth_fn(compress_eval_vals),
        color="b",
        label="Compressed",
    )
    ax[1].set_xlabel("Iteration", fontsize=14)
    ax[1].set_ylabel("Pearson Corr.", fontsize=14)
    ax[1].legend(fontsize=12)
    return fig


if __name__ == "__main__":
    main()

    fig = plot_results()
    fig.savefig("figures/deep_lora_narrow_vs_wide.png", bbox_inches="tight", dpi=500)
