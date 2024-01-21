import os

import flax
import flax.struct
import flax.training.checkpoints
import jax

import configs


def write_train_metric(summary_writer, train_metrics, step):
    for i, metric in enumerate(train_metrics):
        for key, val in metric.items():
            tag = f"train_{key}"
            summary_writer.scalar(tag, val, step - len(train_metrics) + i + 1)


def write_eval_metric(summary_writer, eval_metrics, step):
    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)


def save_lora_state(
    experiment_name: str, step: int, lora_params: flax.core.FrozenDict[str, jax.Array]
):
    flax.training.checkpoints.save_checkpoint(
        ckpt_dir=os.path.join(
            "/home/ubuntu/deep-lora-transformers",
            experiment_name,
            "checkpoints",
        ),
        step=step,
        keep=20,
        target=lora_params,
    )


def load_lora_state(experiment_name: str, step: int):
    return flax.training.checkpoints.restore_checkpoint(
        os.path.join(
            "/home/ubuntu/deep-lora-transformers",
            experiment_name,
            f"checkpoints/checkpoint_{step}",
        ),
        target=None,
    )


def get_task_config_from_json(experiment_dir: str):
    with open(os.path.join(experiment_dir, "config.json"), "r") as f:
        return configs.TaskConfig.from_json(f.read())  # type: ignore
