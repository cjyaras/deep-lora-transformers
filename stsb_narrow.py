import numpy as np

import configs
import data
from finetune import finetune


def common_config():
    task_config = configs.TaskConfig()
    task_config.finetune_task_name = "stsb"
    task_config.num_train_samples = 1024
    task_config.train_batch_size = 16
    task_config.max_seq_length = 128
    task_config.lora_init_scale = 1e-3
    task_config.num_train_steps = 1000
    task_config.log_eval_steps = 20
    task_config.decay_ratio = 1.0
    task_config.lora_depth = 3
    task_config.learning_rate = 1e-4
    task_config.lora_adapt_type = configs.LoraAdaptType.all_dense
    task_config.save_dir = f"experiments/stsb_fewshot_1024_narrow_vs_wide/"
    return task_config


def run_experiments(rank, seeds):
    task_config = common_config()
    task_config.lora_rank = rank
    finetune(task_config, seeds)


def main():
    seeds = [0]
    # run_experiments(rank=8, seeds=seeds)
    run_experiments(rank=None, seeds=seeds)


if __name__ == "__main__":
    main()
