import numpy as np

import configs
from finetune import finetune


def common_config():
    task_config = configs.TaskConfig()
    task_config.train_batch_size = 16
    task_config.max_seq_length = 128
    task_config.lora_init_scale = 1e-3
    task_config.decay_ratio = 1.0
    task_config.lora_adapt_type = configs.LoraAdaptType.all_dense
    return task_config


def run_experiments(num_samples, depth, rank, learning_rate, seeds):
    task_config = common_config()
    task_config.num_train_samples = num_samples
    task_config.num_train_steps = 500
    task_config.save_step_points = [500]
    task_config.log_eval_steps = task_config.num_train_steps
    task_config.finetune_task_name = "stsb"
    task_config.save_dir = f"experiments/stsb_256_checkpoints"
    task_config.lora_depth = depth
    task_config.lora_rank = rank
    task_config.learning_rate = learning_rate

    finetune(task_config, seeds)


def main():
    seeds = [0]
    num_samples = 256
    run_experiments(num_samples, depth=2, rank=8, learning_rate=1e-4, seeds=seeds)
    run_experiments(num_samples, depth=3, rank=None, learning_rate=1e-4, seeds=seeds)


if __name__ == "__main__":
    main()
