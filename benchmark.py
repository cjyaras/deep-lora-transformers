import configs
from finetune import finetune


def main():
    task_config = configs.TaskConfig()
    task_config.num_train_steps = 2000
    task_config.decay_ratio = 0.1
    task_config.log_eval_steps = 200
    finetune(task_config)

    task_config.lora_depth = 2
    task_config.lora_rank = 8
    finetune(task_config)


if __name__ == "__main__":
    main()
