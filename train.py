from functools import partial
from typing import Callable, Tuple

import flax
import flax.struct
import flax.training.common_utils as flax_cu
import flax.training.train_state as train_state
import jax
import jax.numpy as jnp
import optax
import transformers

import configs
import models


def create_learning_rate_fn(
    num_train_steps: int,
    num_warmup_steps: int,
    learning_rate: float,
    decay_ratio: float,
) -> optax.Schedule:
    """Returns a linear warmup, linear decay learning rate function."""
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


def create_train_eval_step_fns(
    learning_rate_fn: optax.Schedule, use_lora: bool
) -> Tuple[Callable, Callable]:
    if not use_lora:

        @partial(jax.jit, donate_argnums=(0,))
        def train_step(
            model_state: ModelTrainState,
            batch: dict[str, jax.Array],
            dropout_rng: jax.Array,
        ):
            dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
            targets = batch.pop("labels")

            def loss_fn(params):
                logits = model_state.apply_fn(
                    **batch, params=params, dropout_rng=dropout_rng, train=True
                )[0]
                loss = model_state.loss_fn(logits, targets)
                return loss

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(model_state.params)
            new_model_state = model_state.apply_gradients(grads=grads)
            metrics = {
                "loss": loss,
                "learning_rate": learning_rate_fn(model_state.step),
            }
            return new_model_state, metrics, new_dropout_rng

        @jax.jit
        def eval_step(model_state: ModelTrainState, batch: dict[str, jax.Array]):
            logits = model_state.apply_fn(
                **batch, params=model_state.params, train=False
            )[0]
            return model_state.logits_fn(logits)

    else:

        @partial(jax.jit, donate_argnums=(1,))
        def train_step(
            model_state: ModelTrainState,
            lora_state: LoraTrainState,
            batch: dict[str, jax.Array],
            dropout_rng: jax.Array,
        ):
            dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
            targets = batch.pop("labels")

            def loss_fn(lora_params):
                adapted_model_params = lora_state.apply_fn(
                    {"params": lora_params}, model_state.params
                )
                logits = model_state.apply_fn(
                    **batch,
                    params=adapted_model_params,
                    dropout_rng=dropout_rng,
                    train=True,
                )[0]
                loss = model_state.loss_fn(logits, targets)
                return loss

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(lora_state.params)
            new_lora_state = lora_state.apply_gradients(grads=grads)
            metrics = {"loss": loss, "learning_rate": learning_rate_fn(lora_state.step)}
            return new_lora_state, metrics, new_dropout_rng

        @jax.jit
        def eval_step(
            model_state: ModelTrainState,
            lora_state: LoraTrainState,
            batch: dict[str, jax.Array],
        ):
            adapted_model_params = lora_state.apply_fn(
                {"params": lora_state.params}, model_state.params
            )
            logits = model_state.apply_fn(
                **batch, params=adapted_model_params, train=False
            )[0]
            return model_state.logits_fn(logits)

    return train_step, eval_step


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
            xentropy = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
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
    task_config: configs.TaskConfig,
    model_params: flax.core.FrozenDict[str, jax.Array],
    learning_rate_fn: optax.Schedule,
    lora_rng: jax.Array,
) -> LoraTrainState:
    lora_model = models.create_lora_model_from_config(task_config, model_params)
    lora_variables = lora_model.init(lora_rng)
    lora_params = lora_variables["params"]
    tx = optax.adamw(
        learning_rate=learning_rate_fn,
        b1=0.9,
        b2=0.999,
        eps=1e-6,
        weight_decay=task_config.weight_decay,
    )
    return LoraTrainState.create(
        apply_fn=partial(lora_model.apply, method=lora_model.adapt),
        params=lora_params,
        tx=tx,
    )


def write_train_metric(summary_writer, train_metrics, step):
    for i, metric in enumerate(train_metrics):
        for key, val in metric.items():
            tag = f"train_{key}"
            summary_writer.scalar(tag, val, step - len(train_metrics) + i + 1)


def write_eval_metric(summary_writer, eval_metrics, step):
    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)
