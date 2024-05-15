from typing import Optional

import jax.numpy as jnp
import numpy as np
import optax
from jax import Array


def ce_loss(logits: Array, labels: np.ndarray, padding: Optional[np.ndarray] = None):
    batch_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    if padding is not None:
        batch_loss = batch_loss * padding
        return jnp.sum(batch_loss) / jnp.sum(padding)
    else:
        return jnp.mean(batch_loss)


def mse_loss(logits: Array, labels: np.ndarray):
    return jnp.mean((logits[..., 0] - labels) ** 2)


class EvalMetric:
    pass


# metric = evaluate.load("rouge", cache_dir=model_args.cache_dir)


# def postprocess_text(preds, labels):
#     preds = [pred.strip() for pred in preds]
#     labels = [label.strip() for label in labels]

#     # rougeLSum expects newline after each sentence
#     preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
#     labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

#     return preds, labels


# def compute_metrics(preds, labels):
#     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     # Some simple post-processing
#     decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

#     result = metric.compute(
#         predictions=decoded_preds, references=decoded_labels, use_stemmer=True
#     )
#     result = {k: round(v * 100, 4) for k, v in result.items()}
#     prediction_lens = [
#         np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
#     ]
#     result["gen_len"] = np.mean(prediction_lens)
#     return result
