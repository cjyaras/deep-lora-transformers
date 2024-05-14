import os

import chex
import flax
import flax.training.checkpoints
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np

import configs


def shift_tokens_right(
    input_ids: np.ndarray, pad_token_id: int, decoder_start_token_id: int
):
    "Shift input ids one token to the right."
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id

    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids[shifted_input_ids == -100] = pad_token_id

    return shifted_input_ids


def cosine_angle(U, V):
    "Computes the cosine subspace angle between two subspaces spanned by the columns of U and V."
    return np.linalg.norm(np.linalg.svd(U.T @ V, compute_uv=False), ord=np.inf)  # type: ignore


def pad_to_batch_size(batch: dict[str, np.ndarray], target_batch_size: int):
    "Pads batch input to target batch size, also returns original length label."
    labels = batch.pop("labels")
    padded_batch = jax.tree_util.tree_map(
        lambda x: np.pad(
            x, ((0, target_batch_size - len(labels)), (0, 0))  # type: ignore
        ),  # type: ignore
        batch,
    )
    return padded_batch, labels


# TODO: Need to do generic name check for parameters (e.g. "kernel" in name) for BERT/BART


def get_filtered_flat_params_shape_dict(
    model_params: chex.ArrayTree, lora_adapt_type: configs.LoraAdaptType
):
    flat_params = flax.traverse_util.flatten_dict(model_params, sep="/")
    flat_params_shape_dict = jax.tree_util.tree_map(jnp.shape, flat_params)
    if lora_adapt_type == configs.LoraAdaptType.ONLY_QUERY_VALUE:
        filter_fn = (
            lambda flat_path, _: "query/kernel" in flat_path
            or "value/kernel" in flat_path
        )
    elif lora_adapt_type == configs.LoraAdaptType.ATTENTION_MLP:
        filter_fn = (
            lambda flat_path, _: "query/kernel" in flat_path
            or "key/kernel" in flat_path
            or "value/kernel" in flat_path
            or "intermediate/dense/kernel" in flat_path
            or "output/dense/kernel" in flat_path
        )
    else:
        filter_fn = lambda _, shape: len(shape) == 2 and min(shape) >= 768
    return {
        name: shape
        for name, shape in flat_params_shape_dict.items()
        if filter_fn(name, shape)
    }


# Training/logging functions


def write_train_metric(summary_writer, result_dict, train_metrics, step):
    for i, metric in enumerate(train_metrics):
        for key, val in metric.items():
            tag = f"train_{key}"
            if tag not in result_dict:
                result_dict[tag] = []
            result_dict[tag].append(
                {"step": step - len(train_metrics) + i + 1, "value": val.item()}
            )
            summary_writer.scalar(tag, val, step - len(train_metrics) + i + 1)


def write_eval_metric(summary_writer, result_dict, eval_metrics, step):
    for metric_name, value in eval_metrics.items():
        tag = f"eval_{metric_name}"
        if tag not in result_dict:
            result_dict[tag] = []
        result_dict[tag].append({"step": step, "value": value})
        summary_writer.scalar(f"eval_{metric_name}", value, step)


def save_lora_params(experiment_path: str, step: int, lora_params: chex.ArrayTree):
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


def get_experiment_name(task_config: configs.TaskConfig, seed: int):
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


def get_experiment_path(task_config: configs.TaskConfig, seed: int):
    experiment_name = get_experiment_name(task_config, seed)
    return os.path.join(os.getcwd(), task_config.save_dir, experiment_name)
