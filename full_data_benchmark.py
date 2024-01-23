import configs
from finetune import finetune


def main():
    task_config = configs.TaskConfig()
    task_config.pretrain_model = "roberta-base"
    task_config.num_train_steps = 2000
    task_config.log_eval_steps = 200
    task_config.lora_depth = 2
    task_config.lora_init_scale = 1e-5
    task_config.lora_rank = 8
    task_config.learning_rate = 5e-4
    task_config.decay_ratio = 0.1
    finetune(task_config)

    task_config.lora_depth = 3
    task_config.lora_rank = None
    task_config.lora_init_scale = 1e-3
    task_config.learning_rate = 5e-5
    finetune(task_config)


if __name__ == "__main__":
    main()
