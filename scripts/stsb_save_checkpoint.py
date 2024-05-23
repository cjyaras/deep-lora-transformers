import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm.auto import tqdm

from dlt import configs, logging_utils, models
from dlt.finetune import finetune


def common_config():
    task_config = configs.TaskConfig()
    task_config.train_batch_size = 16
    task_config.max_seq_length = 128
    task_config.lora_init_scale = 1e-3
    task_config.decay_ratio = 1.0
    task_config.lora_adapt_type = configs.LoraAdaptType.ALL_DENSE
    return task_config


def run_experiments(num_samples, depth, rank, learning_rate, seeds):
    task_config = common_config()
    task_config.num_train_samples = num_samples
    task_config.num_train_steps = 500
    task_config.save_step_points = [500]
    task_config.log_eval_steps = task_config.num_train_steps
    task_config.finetune_task_name = configs.GlueTaskName.STSB
    task_config.save_dir = f"checkpoints/stsb_256_checkpoints"
    task_config.lora_depth = depth
    task_config.lora_rank = rank
    task_config.learning_rate = learning_rate

    finetune(task_config, seeds)


def main():
    seeds = [0]
    num_samples = 256
    run_experiments(num_samples, depth=2, rank=8, learning_rate=1e-4, seeds=seeds)
    run_experiments(num_samples, depth=3, rank=None, learning_rate=1e-4, seeds=seeds)


def get_results():
    experiment_dir = "checkpoints/stsb_256_checkpoints"
    runs = os.listdir(experiment_dir)
    run_2 = [run for run in runs if "depth=2" in run][0]
    run_3 = [run for run in runs if "depth=3" in run][0]
    task_config_2 = logging_utils.get_task_config_from_json(
        experiment_path=os.path.join(experiment_dir, run_2)
    )
    task_config_3 = logging_utils.get_task_config_from_json(
        experiment_path=os.path.join(experiment_dir, run_3)
    )
    model_params = models.create_pretrain_model_from_config(task_config_2).params  # type: ignore
    lora_model_2 = models.create_lora_model_from_config(task_config_2, model_params)
    lora_model_3 = models.create_lora_model_from_config(task_config_3, model_params)
    run_2_e2e = lora_model_2.apply(
        {
            "params": logging_utils.load_lora_params(
                os.path.join(experiment_dir, run_2), 500
            )  # type: ignore
        }
    )
    run_3_e2e = lora_model_3.apply(
        {
            "params": logging_utils.load_lora_params(
                os.path.join(experiment_dir, run_3), 500
            )  # type: ignore
        }
    )
    run_2_ranks_norms = [(np.linalg.matrix_rank(v), np.linalg.norm(v, ord=2)) for v in tqdm(run_2_e2e.values())]  # type: ignore
    run_3_ranks_norms = [(np.linalg.matrix_rank(v), np.linalg.norm(v, ord=2)) for v in tqdm(run_3_e2e.values())]  # type: ignore
    run_2_ranks = [x[0] for x in run_2_ranks_norms]
    run_3_ranks = [min(x[0], 8) if x[1] > 1e-3 else 0 for x in run_3_ranks_norms]  # type: ignore
    return run_2_ranks, run_3_ranks


def plot_results():
    run_2_ranks, run_3_ranks = get_results()
    fig, ax = plt.subplots()
    sns.histplot(run_2_ranks, ax=ax, color="orange", discrete=True)
    sns.histplot(run_3_ranks, ax=ax, color="blue", discrete=True)
    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(["Vanilla LoRA", "Deep LoRA"], fontsize=16)
    return fig


if __name__ == "__main__":
    main()

    fig = plot_results()
    fig.savefig("figures/fewshot_256_ranks.png", bbox_inches="tight", dpi=500)
