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
    task_config.save_dir = f"experiments/stsb_fewshot_16_varying_rank/{rank}"
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


if __name__ == "__main__":
    main()
