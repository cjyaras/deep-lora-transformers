from functools import partial
from typing import Callable, Tuple

import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
from chex import ArrayTree
from flax.training.train_state import TrainState
from jax import Array
from optax import GradientTransformation, Schedule
from tqdm.auto import tqdm
from transformers import (
    FlaxAutoModel,
    FlaxBartForConditionalGeneration,
    FlaxT5ForConditionalGeneration,
    GenerationConfig,
)

from . import metrics, model_utils, models
from .configs import TaskConfig, TaskType
from .models import Lora


def create_learning_rate_fn(
    num_train_steps: int,
    num_warmup_steps: int,
    learning_rate: float,
    decay_ratio: float,
) -> Schedule:
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


def create_optimizer(
    learning_rate_fn: Schedule, weight_decay: float
) -> GradientTransformation:
    return optax.adamw(
        learning_rate=learning_rate_fn,
        b1=0.9,
        b2=0.999,
        eps=1e-6,
        weight_decay=weight_decay,
    )


ModelState = TrainState


class LoraState(TrainState):
    dropout_rng: Array


def create_loss_fn(
    model_state: ModelState,
    lora_state: LoraState,
    is_regression: bool,
    is_train: bool,
) -> Callable:
    "Returns function that computes loss for model/lora state."

    def loss_fn(lora_params: ArrayTree, batch: dict[str, np.ndarray]) -> Array:
        labels = batch.pop("labels")
        adapted_model_params = lora_state.apply_fn(
            {"params": lora_params}, model_state.params
        )
        logits = model_state.apply_fn(
            **batch,
            params=adapted_model_params,
            dropout_rng=lora_state.dropout_rng,
            train=is_train,
        )[0]
        if is_regression:
            loss = metrics.mse_loss(logits, labels)
        else:
            if "decoder_attention_mask" in batch:
                loss = metrics.ce_loss(
                    logits, labels, padding=batch["decoder_attention_mask"]
                )
            else:
                loss = metrics.ce_loss(logits, labels)
        return loss

    return loss_fn


def create_train_step_fn(
    task_config: TaskConfig, learning_rate_fn: optax.Schedule
) -> Callable:

    is_regression = task_config.finetune_task_name == "stsb"

    @partial(jax.jit, donate_argnums=(1,))
    def train_step_fn(
        model_state: ModelState, lora_state: LoraState, batch: dict[str, np.ndarray]
    ) -> tuple[LoraState, dict[str, Array]]:
        _, new_dropout_rng = jax.random.split(lora_state.dropout_rng)
        loss_and_grad_fn = jax.value_and_grad(
            create_loss_fn(model_state, lora_state, is_regression, is_train=True)
        )
        loss, grads = loss_and_grad_fn(lora_state.params, batch)
        new_lora_state = lora_state.apply_gradients(grads=grads)
        new_lora_state = new_lora_state.replace(dropout_rng=new_dropout_rng)
        metrics = {"loss": loss, "learning_rate": learning_rate_fn(lora_state.step)}
        return new_lora_state, metrics

    return train_step_fn


def create_validate_step_fn(task_config: TaskConfig) -> Callable:

    is_regression = task_config.finetune_task_name == "stsb"

    @jax.jit
    def validate_step_fn(
        model_state: ModelState, lora_state: LoraState, batch: dict[str, np.ndarray]
    ) -> dict[str, Array]:
        loss_fn = create_loss_fn(model_state, lora_state, is_regression, is_train=False)
        loss = loss_fn(lora_state.params, batch)
        metrics = {"loss": loss}
        return metrics

    return validate_step_fn


def create_eval_step_fn(task_config: TaskConfig):

    is_regression = task_config.finetune_task_name == "stsb"

    @jax.jit
    def eval_step_fn(
        model_state: ModelState, lora_state: LoraState, batch: dict[str, Array]
    ):
        adapted_model_params = lora_state.apply_fn(
            {"params": lora_state.params}, model_state.params
        )
        logits = model_state.apply_fn(
            **batch, params=adapted_model_params, train=False
        )[0]

        return logits[..., 0] if is_regression else logits.argmax(-1)

    return eval_step_fn


def create_decode_step_fn(model: FlaxAutoModel, task_config: TaskConfig) -> Callable:

    assert isinstance(
        model, FlaxT5ForConditionalGeneration
    ), "Only T5 supports decoding."
    assert isinstance(
        task_config.max_seq_length, Tuple
    ), "Tuple expected for max_seq_length."
    assert (
        task_config.task_type == TaskType.NLG
    ), "Only generation tasks supports decoding."

    gen_kwargs = {
        "max_length": task_config.max_seq_length[1],
        "num_beams": 4,
        "decoder_start_token_id": model._get_decoder_start_token_id(),
    }

    @jax.jit
    def decode_step_fn(
        model_state: ModelState,
        lora_state: LoraState,
        batch: dict[str, jax.Array],
    ):
        adapted_model_params = lora_state.apply_fn(
            {"params": lora_state.params}, model_state.params
        )
        output_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            params=adapted_model_params,
            generation_config=GenerationConfig.from_dict(gen_kwargs),
        )
        return output_ids.sequences

    return decode_step_fn


def create_model_state(model: FlaxAutoModel) -> ModelState:
    """Create (frozen) model state."""

    return ModelState.create(
        apply_fn=model.__call__,  # type: ignore
        params=model.params,  # type: ignore
        tx=optax.set_to_zero(),
    )


def create_lora_state(
    task_config: TaskConfig,
    model_params: ArrayTree,
    learning_rate_fn: Schedule,
    lora_rng: Array,
    dropout_rng: Array,
) -> tuple[LoraState, Lora]:
    lora_model = models.create_lora_model_from_config(task_config, model_params)
    lora_variables = lora_model.init(lora_rng)
    lora_params = lora_variables["params"]
    tx = create_optimizer(learning_rate_fn, task_config.weight_decay)
    return (
        LoraState.create(
            apply_fn=partial(lora_model.apply, method=lora_model.adapt),
            params=lora_params,
            tx=tx,
            dropout_rng=dropout_rng,
        ),
        lora_model,
    )


def create_compressed_lora_train_state(
    uncompressed_lora_state: LoraState,
    uncompressed_lora_model: Lora,
    model_state: ModelState,
    batch: dict[str, np.ndarray],
    task_config: TaskConfig,
):
    assert task_config.lora_compress, "Lora compression is not enabled."
    rank = task_config.lora_rank
    assert rank is not None, "Rank must be specified."
    assert rank % 2 == 0, "Rank must be even."
    compressed_lora_model = models.Lora(
        flat_params_shape_dict=model_utils.get_filtered_flat_params_shape_dict(
            model_state.params, task_config.lora_adapt_type
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
    loss_fn = create_loss_fn(
        model_state,
        uncompressed_lora_state,
        task_config.finetune_task_name != "stsb",
        True,
    )

    uncompressed_grads = jax.grad(loss_fn)(uncompressed_lora_state.params, batch)

    # Move to numpy (do compression on CPU to save GPU memory)
    uncompressed_lora_params_numpy = jax.tree_map(
        np.array, uncompressed_lora_state.params
    )
    uncompressed_grads_numpy = jax.tree_map(np.array, uncompressed_grads)
    uncompressed_e2e_numpy = jax.tree_map(np.array, uncompressed_e2e)

    def svd(A):
        U, s, VT = np.linalg.svd(A, full_matrices=True)
        return U, s, VT.T

    def get_left_right_factors(W1, W1_grad, e2e):
        m, n = W1.shape
        swap = m < n
        if swap:
            W1 = W1.T
            W1_grad = W1_grad.T
            m, n = n, m

        half_rank = rank // 2
        Ugrad, _, Vgrad = svd(W1_grad)
        Va = W1.T @ Ugrad[:, half_rank:] / task_config.lora_init_scale
        Vb = Vgrad[:, half_rank:]
        V0 = Va @ svd(np.concatenate([Va, -Vb], axis=1))[2][:half_rank, n:]
        V = svd(V0)[0][:, ::-1]
        right = V[:, :rank]
        left = e2e @ right / (task_config.lora_init_scale**task_config.lora_depth)

        if swap:
            return right, left
        else:
            return left, right

    compressed_lora_params_numpy = {}

    pbar = tqdm(uncompressed_grads_numpy.items())
    pbar.set_description("Compressing LoRA parameters")

    for k, g in pbar:
        comp_mf_params = {}
        if not task_config.lora_random_factors:
            left, right = get_left_right_factors(
                uncompressed_lora_params_numpy[k]["W1"],
                g["W1"],
                uncompressed_e2e_numpy[k],
            )
            comp_mf_params["left"] = left
            comp_mf_params["right"] = right
        else:
            m, n = uncompressed_e2e_numpy[k].shape
            left = np.random.randn(m, rank)
            left /= np.linalg.norm(left, axis=0, keepdims=True)
            right = np.random.randn(n, rank)
            right /= np.linalg.norm(right, axis=0, keepdims=True)
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

    return LoraState.create(
        apply_fn=partial(
            compressed_lora_model.apply, method=compressed_lora_model.adapt
        ),
        params=compressed_lora_params,
        tx=tx,
    )
