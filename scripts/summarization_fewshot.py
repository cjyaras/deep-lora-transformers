import numpy as np

from dlt import configs
from dlt.data import SUMMARIZATION_TASK_TO_KEYS
from dlt.finetune import finetune


def common_config(num_train_steps):
    task_config = configs.TaskConfig()
    task_config.task_type = configs.TaskType.SUMMARIZATION
    task_config.pretrain_model = configs.ModelType.BART
    task_config.num_train_samples = 1024
    task_config.train_batch_size = 16
    task_config.max_seq_length = (512, 128)
    task_config.lora_init_scale = 1e-3
    task_config.num_train_steps = num_train_steps
    task_config.log_eval_steps = num_train_steps
    task_config.decay_ratio = 1.0
    task_config.lora_adapt_type = configs.LoraAdaptType.ALL_DENSE
    return task_config


def run_experiments(finetune_task_name, depth, rank, learning_rate, seeds):
    task_config = common_config(1000)
    task_config.finetune_task_name = finetune_task_name
    task_config.save_dir = (
        f"../checkpoints/summarization_fewshot_1024/{finetune_task_name}"
    )
    task_config.lora_depth = depth
    task_config.lora_rank = rank
    task_config.learning_rate = learning_rate

    finetune(task_config, seeds)


def main():
    seeds = np.arange(10)
    for finetune_task_name in list(SUMMARIZATION_TASK_TO_KEYS.keys()):
        run_experiments(
            finetune_task_name,
            depth=2,
            rank=8,
            learning_rate=1e-4,
            seeds=seeds,
        )
        run_experiments(
            finetune_task_name,
            depth=3,
            rank=None,
            learning_rate=1e-4,
            seeds=seeds,
        )


if __name__ == "__main__":
    main()

    # results, labels = get_results()
    # for series, tag in zip(results, labels):
    #     print(tag, np.mean(series), np.var(series))