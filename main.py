import configs
from finetune import finetune


def main():
    task_config = configs.TaskConfig()
    task_config.save_step_points = [1, -1]
    finetune(task_config)


if __name__ == "__main__":
    main()
