import configs
from finetune import finetune


def main():
    task_config = configs.TaskConfig()
    task_config.num_train_steps = 2000
    task_config.log_eval_steps = 100
    task_config.learning_rate = 1e-4
    task_config.decay_ratio = 0.1
    task_config.save_step_points = [
        0,
        1,
        10,
        20,
        30,
        100,
        500,
        1000,
        1500,
        task_config.num_train_steps,
    ]
    finetune(task_config)


if __name__ == "__main__":
    main()
