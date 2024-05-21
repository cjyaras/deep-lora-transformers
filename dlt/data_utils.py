from typing import Tuple

import jax.tree_util
import numpy as np


def shift_tokens_right(
    input_ids: np.ndarray, pad_token_id: int, decoder_start_token_id: int
) -> np.ndarray:
    "Shift input ids one token to the right."
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id

    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids = np.where(
        shifted_input_ids != -100, shifted_input_ids, pad_token_id
    )

    return shifted_input_ids


def pad_to_batch_size(
    batch: dict[str, np.ndarray], target_batch_size: int
) -> Tuple[dict[str, np.ndarray], np.ndarray]:
    "Pads batch input to target batch size, also returns original length label."
    labels = batch.pop("labels")
    padded_batch = jax.tree_util.tree_map(
        lambda x: np.pad(
            x, ((0, target_batch_size - len(labels)), (0, 0))  # type: ignore
        ),  # type: ignore
        batch,
    )
    return padded_batch, labels
