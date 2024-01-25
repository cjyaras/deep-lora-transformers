import configs
import data
from finetune import finetune


def common_config():
    task_config = configs.TaskConfig()
    task_config.num_train_samples = 32
    task_config.train_batch_size = task_config.num_train_samples
    task_config.max_seq_length = 64
    task_config.lora_init_scale = 1
    task_config.num_train_steps = 200
    task_config.log_eval_steps = 200
    task_config.decay_ratio = 0.1
    task_config.lora_adapt_type = configs.LoraAdaptType.only_query_value
    return task_config


def run_experiments(finetune_task_name, depth, rank, learning_rate, seeds):
    task_config = common_config()
    task_config.finetune_task_name = finetune_task_name
    task_config.save_dir = f"experiments/few_shot/{finetune_task_name}"
    task_config.lora_depth = depth
    task_config.lora_rank = rank
    task_config.learning_rate = learning_rate

    finetune(task_config, seeds)


def main():
    seeds = [0, 1, 2, 3, 4]
    for finetune_task_name in data.task_to_keys.keys():
        run_experiments(
            finetune_task_name, depth=2, rank=8, learning_rate=5e-4, seeds=seeds
        )
        run_experiments(
            finetune_task_name, depth=3, rank=None, learning_rate=1e-4, seeds=seeds
        )


if __name__ == "__main__":
    main()
