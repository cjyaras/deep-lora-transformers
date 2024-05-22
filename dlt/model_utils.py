from typing import Tuple

import flax.traverse_util
import jax
import jax.numpy as jnp
from chex import ArrayTree

from .configs import LoraAdaptType

BERT_ONLY_QUERY_VALUE = [
    "attention/self/query/kernel",
    "attention/self/value/kernel",
]
BERT_ATTENTION_MLP = [
    "attention/self/query/kernel",
    "attention/self/key/kernel",
    "attention/self/value/kernel",
    "attention/output/dense/kernel",
    "intermediate/dense/kernel",
    "output/dense/kernel",
]
T5_ONLY_QUERY_VALUE = [
    "EncDecAttention/q/kernel",
    "EncDecAttention/v/kernel",
    "SelfAttention/q/kernel",
    "SelfAttention/v/kernel",
]
T5_ATTENTION_MLP = [
    "EncDecAttention/q/kernel",
    "EncDecAttention/k/kernel",
    "EncDecAttention/v/kernel",
    "EncDecAttention/o/kernel",
    "SelfAttention/q/kernel",
    "SelfAttention/k/kernel",
    "SelfAttention/v/kernel",
    "SelfAttention/o/kernel",
    "DenseReluDense/wi/kernel",
    "DenseReluDense/wo/kernel",
]

MIN_DIMS = 512


def is_path_valid(path: str, valid_list: list) -> bool:
    "Checks whether path contains any of the valid paths in the valid_list."
    return any([valid_path in path for valid_path in valid_list])


def get_filtered_flat_params_shape_dict(
    model_params: ArrayTree, lora_adapt_type: LoraAdaptType
) -> dict[str, Tuple[int, int]]:
    "Returns a dictionary of parameter names to adapt and their shapes."
    flat_params = flax.traverse_util.flatten_dict(model_params, sep="/")
    flat_params_shape_dict = jax.tree_util.tree_map(jnp.shape, flat_params)
    if lora_adapt_type == LoraAdaptType.ONLY_QUERY_VALUE:
        filter_fn = lambda flat_path, _: is_path_valid(
            flat_path,
            BERT_ONLY_QUERY_VALUE + T5_ONLY_QUERY_VALUE,
        )
    elif lora_adapt_type == LoraAdaptType.ATTENTION_MLP:
        filter_fn = lambda flat_path, _: is_path_valid(
            flat_path, BERT_ATTENTION_MLP + T5_ATTENTION_MLP
        )
    else:
        filter_fn = lambda _, shape: len(shape) == 2 and min(shape) >= MIN_DIMS
    return {
        name: shape
        for name, shape in flat_params_shape_dict.items()
        if filter_fn(name, shape)
    }
