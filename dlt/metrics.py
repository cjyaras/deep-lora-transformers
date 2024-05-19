from typing import Dict, List, Optional, Tuple

import evaluate
import jax.numpy as jnp
import nltk
import numpy as np
import optax
import transformers
from jax import Array

from .configs import GlueTaskName, ModelType

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def ce_loss(logits: Array, labels: np.ndarray, padding: Optional[np.ndarray] = None):
    batch_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    if padding is not None:
        batch_loss = batch_loss * padding
        return jnp.sum(batch_loss) / jnp.sum(padding)
    else:
        return jnp.mean(batch_loss)


def mse_loss(logits: Array, labels: np.ndarray):
    return jnp.mean((logits[..., 0] - labels) ** 2)


class GlueEvalMetric:

    def __init__(self, finetune_task_name: GlueTaskName):
        self.eval_metric = evaluate.load("glue", finetune_task_name)

    def add_batch(self, predictions, references):
        self.eval_metric.add_batch(predictions=predictions, references=references)

    def compute(self) -> Dict:
        result = self.eval_metric.compute()
        assert result is not None
        result = {k: round(v * 100, 2) for k, v in result.items()}
        return result


class SummarizationEvalMetric:

    def __init__(self, pretrain_model: ModelType):
        assert (
            pretrain_model == ModelType.BART
        ), "Only BART is supported for summarization"
        self.eval_metric = evaluate.load("rouge")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(pretrain_model)

    @staticmethod
    def postprocess_text(
        preds: List[str], labels: List[str]
    ) -> Tuple[List[str], List[str]]:
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def add_batch(self, predictions, references):
        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        decoded_labels = self.tokenizer.batch_decode(
            references, skip_special_tokens=True
        )
        decoded_preds, decoded_labels = self.postprocess_text(
            decoded_preds, decoded_labels
        )
        self.eval_metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    def compute(self) -> Dict:
        result = self.eval_metric.compute()
        assert result is not None
        result = {k: round(v * 100, 2) for k, v in result.items()}
        return result
