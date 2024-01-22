import configs
from finetune import finetune


def main():
    task_config = configs.TaskConfig()
    task_config.save_step_points = [
        0,
        1,
        2,
        4,
        8,
        16,
        20,
        30,
        task_config.num_train_steps,
    ]
    finetune(task_config)


if __name__ == "__main__":
    main()
