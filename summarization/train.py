from functools import partial
from typing import Callable, Tuple

import configs
import flax
import flax.core
import flax.struct
import flax.training.checkpoints
import flax.training.train_state as train_state
import flax.traverse_util
import jax
import jax.numpy as jnp
import models
import numpy as np
import optax
import transformers
import utils
from tqdm.auto import tqdm

LoraTrainState = train_state.TrainState


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


def create_optimizer(learning_rate_fn: optax.Schedule, weight_decay: float):
    return optax.adamw(
        learning_rate=learning_rate_fn,
        b1=0.9,
        b2=0.999,
        eps=1e-6,
        weight_decay=weight_decay,
    )


def get_grad_fn(model_state, lora_state):
    def loss_fn(lora_params, batch, dropout_rng):
        targets = batch.pop("labels")
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

    return jax.value_and_grad(loss_fn)


def create_train_eval_predict_step_fns(
    learning_rate_fn: optax.Schedule,
) -> Tuple[Callable, Callable, Callable]:
    @partial(jax.jit, donate_argnums=(1,))
    def train_step(
        model_state: ModelState,
        lora_state: LoraTrainState,
        batch: dict[str, jax.Array],
        dropout_rng: jax.Array,
    ):
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
        grad_fn = get_grad_fn(model_state, lora_state)
        loss, grads = grad_fn(lora_state.params, batch, dropout_rng)
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

    @jax.jit
    def predict_step(
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
        return logits

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
) -> ModelState:
    """Create (frozen) model state."""
    if is_regression:

        def mse_loss(logits, labels):
            return jnp.mean((logits[..., 0] - labels) ** 2)

        return ModelState.create(
            apply_fn=model.__call__,  # type: ignore
            params=model.params,  # type: ignore
            tx=optax.set_to_zero(),
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
            tx=optax.set_to_zero(),
            logits_fn=lambda logits: logits.argmax(-1),
            loss_fn=cross_entropy_loss,
        )


def create_lora_train_state(
    task_config: configs.TaskConfig,
    model_params: flax.core.FrozenDict[str, jax.Array],
    learning_rate_fn: optax.Schedule,
    lora_rng: jax.Array,
) -> tuple[LoraTrainState, models.LoRA]:
    lora_model = models.create_lora_model_from_config(task_config, model_params)
    lora_variables = lora_model.init(lora_rng)
    lora_params = lora_variables["params"]
    tx = create_optimizer(learning_rate_fn, task_config.weight_decay)
    return (
        LoraTrainState.create(
            apply_fn=partial(lora_model.apply, method=lora_model.adapt),
            params=lora_params,
            tx=tx,
        ),
        lora_model,
    )


def create_compressed_lora_train_state(
    uncompressed_lora_state: LoraTrainState,
    uncompressed_lora_model: models.LoRA,
    model_state: ModelState,
    batch: dict,
    dropout_rng: jax.Array,
    task_config: configs.TaskConfig,
):
    assert task_config.lora_compress, "Lora compression is not enabled."
    rank = task_config.lora_rank
    assert rank is not None, "Rank must be specified."
    assert rank % 2 == 0, "Rank must be even."
    compressed_lora_model = models.LoRA(
        flat_params_shape_dict=utils.get_filtered_flat_params_shape_dict(
            model_state.params, task_config
        ),
        depth=task_config.lora_depth,
        init_scale=task_config.lora_init_scale,
        rank=rank,
        compressed=True,
    )

    uncompressed_e2e = uncompressed_lora_model.apply(
        {"params": uncompressed_lora_state.params}
    )

    # Get gradient of uncompressed factors
    value_grad_fn = get_grad_fn(model_state, uncompressed_lora_state)
    _, uncompressed_grads = value_grad_fn(
        uncompressed_lora_state.params, batch, dropout_rng
    )

    # move to numpy
    uncompressed_lora_params_numpy = jax.tree_map(
        np.array, uncompressed_lora_state.params
    )
    uncompressed_grads_numpy = jax.tree_map(np.array, uncompressed_grads)
    uncompressed_e2e_numpy = jax.tree_map(np.array, uncompressed_e2e)

    def get_left_right_factors(w0, g_w0, e2e):
        half_rank = rank // 2
        v1 = jnp.linalg.svd(g_w0, full_matrices=False)[2].T[:, :half_rank]
        v2 = jnp.linalg.svd(g_w0.T @ w0, full_matrices=False)[2].T[:, :half_rank]
        rightT = jnp.linalg.svd(np.concatenate([v1, v2], axis=1), full_matrices=False)[
            0
        ]
        left = e2e @ rightT / (task_config.lora_init_scale**task_config.lora_depth)
        return left, rightT

    compressed_lora_params_numpy = {}

    print("Compressing LoRA parameters...")
    for k, g in tqdm(uncompressed_grads_numpy.items()):
        comp_mf_params = {}
        if not task_config.lora_random_factors:
            left, rightT = get_left_right_factors(
                uncompressed_lora_params_numpy[k]["w0"],
                g["w0"],
                uncompressed_e2e_numpy[k],
            )
            comp_mf_params["left"] = left
            comp_mf_params["right"] = rightT.T
        else:
            m, n = uncompressed_e2e_numpy[k].shape
            left = np.random.randn(m, rank)
            left /= np.linalg.norm(left, axis=0, keepdims=True)
            right = np.random.randn(rank, n)
            right /= np.linalg.norm(right, axis=1, keepdims=True)
            comp_mf_params["left"] = left
            comp_mf_params["right"] = right
        mf_params = {}
        for w in g.keys():
            mf_params[w] = task_config.lora_init_scale * jnp.eye(rank)
        comp_mf_params["mf"] = mf_params
        compressed_lora_params_numpy[k] = comp_mf_params

    compressed_lora_params = jax.tree_map(jnp.array, compressed_lora_params_numpy)

    inner_tx = uncompressed_lora_state.tx
    outer_tx = create_optimizer(
        create_learning_rate_fn(
            task_config.num_train_steps,
            task_config.num_warmup_steps,
            task_config.lora_gamma * task_config.learning_rate,
            task_config.decay_ratio,
        ),
        task_config.weight_decay,
    )
    tx = optax.multi_transform(
        {"inner": inner_tx, "outer": outer_tx},
        flax.traverse_util.path_aware_map(
            lambda p, _: "outer" if p[-1] == "left" or p[-1] == "right" else "inner",
            compressed_lora_params,
        ),
    )

    return LoraTrainState.create(
        apply_fn=partial(
            compressed_lora_model.apply, method=compressed_lora_model.adapt
        ),
        params=compressed_lora_params,
        tx=tx,
    )
