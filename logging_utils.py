import os

import flax.training.checkpoints
from chex import ArrayTree
from flax.metrics.tensorboard import SummaryWriter

from configs import TaskConfig


def write_train_metric(
    summary_writer: SummaryWriter,
    result_dict: dict,
    train_metrics: dict,
    step: int,
):
    for i, metric in enumerate(train_metrics):
        for key, val in metric.items():
            tag = f"train_{key}"
            if tag not in result_dict:
                result_dict[tag] = []
            result_dict[tag].append(
                {"step": step - len(train_metrics) + i + 1, "value": val.item()}
            )
            summary_writer.scalar(tag, val, step - len(train_metrics) + i + 1)


def write_eval_metric(
    summary_writer: SummaryWriter,
    result_dict: dict,
    eval_metrics: dict,
    step: int,
):
    for metric_name, value in eval_metrics.items():
        tag = f"eval_{metric_name}"
        if tag not in result_dict:
            result_dict[tag] = []
        result_dict[tag].append({"step": step, "value": value})
        summary_writer.scalar(f"eval_{metric_name}", value, step)


def save_lora_params(experiment_path: str, step: int, lora_params: ArrayTree):
    flax.training.checkpoints.save_checkpoint(
        ckpt_dir=os.path.join(
            experiment_path,
            "checkpoints",
        ),
        step=step,
        keep=20,
        target=lora_params,
    )


def load_lora_params(experiment_path: str, step: int):
    return flax.training.checkpoints.restore_checkpoint(
        os.path.join(
            experiment_path,
            f"checkpoints/checkpoint_{step}",
        ),
        target=None,
    )


def get_task_config_from_json(experiment_path: str):
    with open(os.path.join(experiment_path, "config.json"), "r") as f:
        return configs.TaskConfig.from_json(f.read())  # type: ignore


def get_experiment_name(task_config: TaskConfig, seed: int):
    experiment_name = f"{task_config.finetune_task_name}_lora"
    experiment_name += f"_type={task_config.lora_adapt_type.value}"
    experiment_name += f"_depth={task_config.lora_depth}"
    if task_config.lora_rank is not None:
        experiment_name += f"_rank={task_config.lora_rank}"
    if task_config.lora_compress:
        experiment_name += "_compress"
        if task_config.lora_random_factors:
            experiment_name += "-random"
    if task_config.num_train_samples is not None:
        experiment_name += f"_samples={task_config.num_train_samples}"
    experiment_name += f"_lr={task_config.learning_rate}"
    experiment_name += f"_steps={task_config.num_train_steps}"
    experiment_name += f"_seed={seed}"
    if task_config.identifier is not None:
        experiment_name += f"_{task_config.identifier}"
    return experiment_name


def get_experiment_path(task_config: TaskConfig, seed: int):
    experiment_name = get_experiment_name(task_config, seed)
    return os.path.join(os.getcwd(), task_config.save_dir, experiment_name)
