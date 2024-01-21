from functools import partial
from typing import Callable, Tuple

import flax
import flax.struct
import flax.training.checkpoints
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
    learning_rate_fn: optax.Schedule,
) -> Tuple[Callable, Callable]:
    @partial(jax.jit, donate_argnums=(1,))
    def train_step(
        model_state: ModelState,
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
        model_state: ModelState,
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


class ModelState(train_state.TrainState):
    """Train state with an Optax optimizer.

    Logit and loss functions differ depending on whether the task is classification
    or regression.

    Args:
      logits_fn: Applied to last layer to obtain the logits.
      loss_fn: Function to compute the loss.
    """

    logits_fn: Callable = flax.struct.field(pytree_node=False)
    loss_fn: Callable = flax.struct.field(pytree_node=False)


def create_model_state(
    model: transformers.FlaxAutoModelForSequenceClassification,
    is_regression: bool,
) -> train_state.TrainState:
    """Create (frozen) model state."""
    tx = optax.set_to_zero()
    if is_regression:

        def mse_loss(logits, labels):
            return jnp.mean((logits[..., 0] - labels) ** 2)

        return ModelState.create(
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

        return ModelState.create(
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
