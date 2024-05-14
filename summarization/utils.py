import numpy as np


def shift_tokens_right(
    input_ids: np.ndarray, pad_token_id: int, decoder_start_token_id: int
):
    "Shift input ids one token to the right."
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id

    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids[shifted_input_ids == -100] = pad_token_id
    # shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids
