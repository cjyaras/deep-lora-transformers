import json
import os

import matplotlib.pyplot as plt
import numpy as np

from dlt import configs
from dlt.finetune import finetune


def common_config():
    task_config = configs.TaskConfig()
    task_config.finetune_task_name = configs.GlueTaskName.STSB
    task_config.num_train_samples = 16
    task_config.train_batch_size = 16
    task_config.max_seq_length = 128
    task_config.lora_init_scale = 1e-3
    task_config.num_train_steps = 400
    task_config.log_eval_steps = 20
    task_config.decay_ratio = 1.0
    task_config.lora_gamma = 1e-2
    task_config.lora_adapt_type = configs.LoraAdaptType.ALL_DENSE
    return task_config


def run_experiments(rank, depth, seeds, learning_rate):
    task_config = common_config()
    task_config.lora_depth = depth
    task_config.lora_rank = rank
    task_config.lora_compress = False
    task_config.learning_rate = learning_rate
    task_config.save_dir = f"../checkpoints/stsb_fewshot_16_varying_rank/{rank}"
    finetune(task_config, seeds)


def main():
    seeds = range(5)
    for rank in [8, 16, 32, 64]:
        run_experiments(
            rank=rank,
            depth=2,
            seeds=seeds,
            learning_rate=1e-4,
        )
        run_experiments(
            rank=rank,
            depth=3,
            seeds=seeds,
            learning_rate=1e-4,
        )


def get_results():
    experiment_dir = "../checkpoints/stsb_fewshot_16_varying_rank"
    ranks = [int(x) for x in os.listdir(experiment_dir)]

    diff_dict = {}
    take_last = True

    def get_seed(run):
        return int(run.split("_")[-1].split("=")[1])

    for sample_size in sorted(ranks):
        runs = [
            run for run in os.listdir(os.path.join(experiment_dir, str(sample_size)))
        ]
        runs_2 = [run for run in runs if "depth=2" in run]
        runs_3 = [run for run in runs if "depth=3" in run]
        runs_2_results = np.zeros(len(runs_2))
        runs_3_results = np.zeros(len(runs_3))
        for run in runs:
            seed = get_seed(run)
            with open(
                os.path.join(experiment_dir, str(sample_size), run, "results.json")
            ) as f:
                results = json.load(f)
            values = [
                entry["value"]
                for entry in sorted(results["eval_pearson"], key=lambda x: x["step"])
            ]
            if "depth=2" in run:
                runs_2_results[seed] = values[-1] if take_last else max(values)
            elif "depth=3" in run:
                runs_3_results[seed] = values[-1] if take_last else max(values)
        diff_dict[sample_size] = (
            runs_2_results,
            runs_3_results,
        )

    ranks, eval_values = list(zip(*diff_dict.items()))
    ranks = list(ranks)
    eval_values = list(eval_values)
    depth_2_values = [x[0] for x in eval_values]
    depth_3_values = [x[1] for x in eval_values]
    depth_2_values = np.array(depth_2_values)
    depth_3_values = np.array(depth_3_values)
    return ranks, depth_2_values, depth_3_values


def plot_results():
    (
        ranks,
        depth_2_values,
        depth_3_values,
    ) = get_results()
    depth_2_mean = depth_2_values.mean(axis=1)
    depth_2_std = depth_2_values.std(axis=1)
    depth_3_mean = depth_3_values.mean(axis=1)
    depth_3_std = depth_3_values.std(axis=1)
    fig, ax = plt.subplots()
    ax.plot(ranks, depth_2_mean, label="Vanilla LoRA", color="orange")
    ax.fill_between(
        ranks,
        depth_2_mean - depth_2_std / 2,
        depth_2_mean + depth_2_std / 2,
        alpha=0.1,
        color="orange",
    )
    ax.plot(ranks, depth_3_mean, label="Deep LoRA", color="blue")
    ax.fill_between(
        ranks,
        depth_3_mean - depth_3_std / 2,
        depth_3_mean + depth_3_std / 2,
        alpha=0.1,
        color="blue",
    )
    ax.set_xscale("log")
    ax.set_xticks(ranks, labels=[int(s) for s in ranks], fontsize=14)
    ax.minorticks_off()
    ax.set_xlabel("Rank $r$", fontsize=14)
    ax.set_ylabel("Pearson Correlation", fontsize=14)
    ax.legend(fontsize=16)
    return fig


if __name__ == "__main__":
    main()

    fig = plot_results()
    fig.savefig(
        "../figures/fewshot_stsb_varying_rank.png", bbox_inches="tight", dpi=500
    )
