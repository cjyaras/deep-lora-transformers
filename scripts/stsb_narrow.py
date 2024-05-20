from dlt import configs
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
    task_config.lora_gamma = 0.01
    task_config.lora_adapt_type = configs.LoraAdaptType.ALL_DENSE
    task_config.save_dir = f"../checkpoints/stsb_fewshot_16_narrow_vs_wide"
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
    run_experiments(
        rank=8, seeds=seeds, compress=True, random=False, learning_rate=1e-2
    )


if __name__ == "__main__":
    main()
