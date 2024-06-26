import os
from typing import Dict, List

import flax.training.checkpoints
import orbax.checkpoint as ocp
from chex import ArrayTree
from flax.metrics.tensorboard import SummaryWriter

from .configs import TaskConfig


def write_train_metric(
    summary_writer: SummaryWriter,
    result_dict: Dict,
    train_metrics: List,
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
    result_dict: Dict,
    eval_metrics: Dict,
    step: int,
):
    for metric_name, value in eval_metrics.items():
        tag = f"eval_{metric_name}"
        if tag not in result_dict:
            result_dict[tag] = []
        result_dict[tag].append({"step": step, "value": value})
        summary_writer.scalar(f"eval_{metric_name}", value, step)


class _Checkpointer:

    def __init__(self, experiment_path: str):
        ckpt_dir = os.path.join(experiment_path, "checkpoints")
        options = ocp.CheckpointManagerOptions(enable_async_checkpointing=True)
        self.manager = ocp.CheckpointManager(
            ckpt_dir, options=options, item_handlers=ocp.StandardCheckpointHandler()
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.manager.wait_until_finished()
        self.manager.close()

    def save(self, step: int, lora_params: ArrayTree):
        self.manager.save(step, args=ocp.args.StandardSave(lora_params))  # type: ignore

    def load(self, step: int) -> ArrayTree:
        lora_params = self.manager.restore(step)
        return lora_params


def save_lora_params(experiment_path: str, step: int, lora_params: ArrayTree):
    absolute_experiment_path = os.path.abspath(experiment_path)
    with _Checkpointer(absolute_experiment_path) as checkpointer:
        checkpointer.save(step, lora_params)


def load_lora_params(experiment_path: str, step: int) -> ArrayTree:
    absolute_experiment_path = os.path.abspath(experiment_path)
    with _Checkpointer(absolute_experiment_path) as checkpointer:
        lora_params = checkpointer.load(step)
    return lora_params


def save_lora_params_old(experiment_path: str, step: int, lora_params: ArrayTree):
    "Old function to save lora params. Use Checkpointer instead."
    flax.training.checkpoints.save_checkpoint(
        ckpt_dir=os.path.join(
            experiment_path,
            "checkpoints",
        ),
        step=step,
        keep=20,
        target=lora_params,
    )


def load_lora_params_old(experiment_path: str, step: int):
    "Old function to load lora params. Use Checkpointer instead."
    return flax.training.checkpoints.restore_checkpoint(
        os.path.join(
            experiment_path,
            f"checkpoints/checkpoint_{step}",
        ),
        target=None,
    )


def get_task_config_from_json(experiment_path: str):
    with open(os.path.join(experiment_path, "config.json"), "r") as f:
        return TaskConfig.from_json(f.read())  # type: ignore


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
