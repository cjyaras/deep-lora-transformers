from typing import Callable

import flax
import flax.struct
import flax.training.common_utils as flax_cu
import flax.training.train_state as train_state
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import transformers

import configs
from models import LoRA

Array = jax.Array


def create_learning_rate_fn(
    train_ds_size: int,
    train_batch_size: int,
    num_train_epochs: int,
    num_warmup_steps: int,
    learning_rate: float,
    decay_ratio: float,
) -> optax.Schedule:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps
    )
    decay_fn = optax.linear_schedule(
        init_value=learning_rate,
        end_value=decay_ratio * learning_rate,
        transition_steps=num_train_steps - num_warmup_steps,
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps]
    )
    return schedule_fn


class ModelTrainState(train_state.TrainState):
    """Train state with an Optax optimizer.

    Logit and loss functions differ depending on whether the task is classification
    or regression.

    Args:
      logits_fn: Applied to last layer to obtain the logits.
      loss_fn: Function to compute the loss.
    """

    logits_fn: Callable = flax.struct.field(pytree_node=False)
    loss_fn: Callable = flax.struct.field(pytree_node=False)


def create_model_train_state(
    model: transformers.FlaxAutoModelForSequenceClassification,
    learning_rate_fn: optax.Schedule,
    is_regression: bool,
    num_labels: int,
    weight_decay: float,
    frozen: bool,
) -> train_state.TrainState:
    """Create initial training state."""

    if not frozen:

        def decay_mask_fn(params):
            flat_params = flax.traverse_util.flatten_dict(params)
            layer_norm_candidates = ["layernorm", "layer_norm", "ln"]
            layer_norm_named_params = {
                layer[-2:]
                for layer_norm_name in layer_norm_candidates
                for layer in flat_params.keys()
                if layer_norm_name in "".join(layer).lower()
            }
            flat_mask = {
                path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params)
                for path in flat_params
            }
            return flax.traverse_util.unflatten_dict(flat_mask)

        tx = optax.adamw(
            learning_rate=learning_rate_fn,
            b1=0.9,
            b2=0.999,
            eps=1e-6,
            weight_decay=weight_decay,
            mask=decay_mask_fn,
        )

    else:
        tx = optax.set_to_zero()

    if is_regression:

        def mse_loss(logits, labels):
            return jnp.mean((logits[..., 0] - labels) ** 2)

        return ModelTrainState.create(
            apply_fn=model.__call__,  # type: ignore
            params=model.params,  # type: ignore
            tx=tx,
            logits_fn=lambda logits: logits[..., 0],
            loss_fn=mse_loss,
        )

    else:  # Classification.

        def cross_entropy_loss(logits, labels):
            xentropy = optax.softmax_cross_entropy(
                logits, flax_cu.onehot(labels, num_classes=num_labels)
            )
            return jnp.mean(xentropy)

        return ModelTrainState.create(
            apply_fn=model.__call__,  # type: ignore
            params=model.params,  # type: ignore
            tx=tx,
            logits_fn=lambda logits: logits.argmax(-1),
            loss_fn=cross_entropy_loss,
        )


class LoraTrainState(train_state.TrainState):
    pass


def create_lora_train_state(
    model_args: configs.ModelArguments,
    model_params: flax.core.FrozenDict[str, Array],
    depth: int,
    learning_rate_fn: optax.Schedule,
    weight_decay: float,
    seed: int = 0,
):
    flat_model_params = flax.traverse_util.flatten_dict(model_params)
    lora_model = LoRA(
        flat_params_keys=[
            k
            for k in flat_model_params
            if k[-2:] == ("query", "kernel") or k[-2:] == ("value", "kernel")
        ],
        depth=depth,
        init_scale=model_args.lora_init_scale,
        inner_dims=model_args.lora_rank,
    )
    lora_variables = lora_model.init(jr.PRNGKey(seed), model_params)
    lora_params = lora_variables["params"]
    tx = optax.adamw(
        learning_rate=learning_rate_fn,
        b1=0.9,
        b2=0.999,
        eps=1e-6,
        weight_decay=weight_decay,
    )
    return LoraTrainState.create(
        apply_fn=lora_model.apply,
        params=lora_params,
        tx=tx,
    )


def write_train_metric(summary_writer, train_metrics, step):
    train_metrics = flax_cu.get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)


def write_eval_metric(summary_writer, eval_metrics, step):
    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)
