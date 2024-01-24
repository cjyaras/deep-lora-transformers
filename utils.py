import os

import flax
import flax.struct
import flax.training.checkpoints
import jax
import jax.numpy as jnp
import numpy as np

import configs

# Misc functions


def svd(x):
    U, s, VT = np.linalg.svd(x)
    return U, s, VT.T


def cosine_angle(U, V):
    return np.linalg.norm(np.linalg.svd(U.T @ V, compute_uv=False), ord=np.inf)


def pad_to_batch_size(batch, target_batch_size):
    labels = batch.pop("labels")
    padded_batch = jax.tree_util.tree_map(
        lambda x: np.pad(
            x, ((0, target_batch_size - len(labels)), (0, 0))  # type: ignore
        ),  # type: ignore
        batch,
    )
    return padded_batch, labels


def get_filtered_flat_params_shape_dict(model_params, task_config):
    flat_params = flax.traverse_util.flatten_dict(model_params, sep="/")
    flat_params_shape_dict = jax.tree_util.tree_map(jnp.shape, flat_params)
    if task_config.lora_adapt_type == configs.LoraAdaptType.only_query_value:
        filter_fn = (
            lambda flat_path, _: "query/kernel" in flat_path
            or "value/kernel" in flat_path
        )
    else:
        filter_fn = (
            lambda flat_path, _: "query/kernel" in flat_path
            or "key/kernel" in flat_path
            or "value/kernel" in flat_path
            or "intermediate/dense/kernel" in flat_path
            or "output/dense/kernel" in flat_path
        )
    return {
        name: shape
        for name, shape in flat_params_shape_dict.items()
        if filter_fn(name, shape)
    }


# Training/logging functions


def write_train_metric(summary_writer, train_metrics, step):
    for i, metric in enumerate(train_metrics):
        for key, val in metric.items():
            tag = f"train_{key}"
            summary_writer.scalar(tag, val, step - len(train_metrics) + i + 1)


def write_eval_metric(summary_writer, eval_metrics, step):
    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)


def save_lora_params(
    experiment_path: str, step: int, lora_params: flax.core.FrozenDict[str, jax.Array]
):
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


def get_experiment_name(task_config: configs.TaskConfig):
    experiment_name = f"{task_config.finetune_task_name}_lora"
    experiment_name += f"_depth={task_config.lora_depth}"
    if task_config.lora_rank is not None:
        experiment_name += f"_rank={task_config.lora_rank}"
    if task_config.num_train_samples is not None:
        experiment_name += f"_samples={task_config.num_train_samples}"
    if task_config.identifier is not None:
        experiment_name += f"_{task_config.identifier}"
    return experiment_name


def get_experiment_path(task_config: configs.TaskConfig):
    experiment_name = get_experiment_name(task_config)
    return os.path.join(os.getcwd(), "experiments", experiment_name)
