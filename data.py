import math
from typing import Optional

import datasets
import flax.training.common_utils as flax_cu
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import transformers

import configs


def load_dataset_from_config(task_config: configs.TaskConfig):
    tokenizer = transformers.AutoTokenizer.from_pretrained(configs.model_type)
    train_dataset, eval_dataset, num_labels, is_regression = load_dataset(
        task_config.finetune_task_name,
        tokenizer,
        task_config.max_seq_length,
        task_config.num_train_samples,
        jax.random.PRNGKey(task_config.sample_seed),
        exclude_long_seq=True,
    )
    return train_dataset, eval_dataset, num_labels, is_regression


def load_dataset(
    finetune_task_name: str,
    tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
    max_seq_length: int,
    num_train_samples: Optional[int],
    sample_rng: jax.Array,
    exclude_long_seq: bool,
):
    raw_datasets = datasets.load_dataset("glue", finetune_task_name)
    is_regression = finetune_task_name == "stsb"

    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names  # type: ignore
        num_labels = len(label_list)
    else:
        num_labels = 1

    def length_of(examples):
        texts = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        return len(tokenizer(*texts)["input_ids"])  # type: ignore

    # Preprocess dataset
    sentence1_key, sentence2_key = configs.task_to_keys[finetune_task_name]

    if exclude_long_seq:
        raw_datasets["train"] = raw_datasets["train"].filter(  # type: ignore
            lambda example: length_of(example) <= max_seq_length
        )

    if num_train_samples is not None:
        raw_datasets["train"] = raw_datasets["train"].select(  # type: ignore
            jr.choice(
                sample_rng,
                jnp.arange(len(raw_datasets["train"])),  # type: ignore
                shape=(num_train_samples,),
            )
        )

    def preprocess_fn(example):
        texts = (
            (example[sentence1_key],)
            if sentence2_key is None
            else (example[sentence1_key], example[sentence2_key])
        )
        result = tokenizer(
            *texts, padding="max_length", max_length=max_seq_length, truncation=True
        )
        result["labels"] = example["label"]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_fn, batched=True, remove_columns=raw_datasets["train"].column_names  # type: ignore
    )

    train_dataset = processed_datasets["train"]  # type: ignore
    eval_dataset = processed_datasets[  # type: ignore
        "validation_matched" if finetune_task_name == "mnli" else "validation"
    ]
    return train_dataset, eval_dataset, num_labels, is_regression


def glue_train_data_collator(
    rng: jr.PRNGKeyArray, dataset: datasets.arrow_dataset.Dataset, batch_size: int
):
    """Returns shuffled batches of size `batch_size` from truncated `train dataset`, sharded over all local devices."""
    steps_per_epoch = len(dataset) // batch_size
    perms = jr.permutation(rng, len(dataset))
    perms = perms[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    perms = perms.reshape((steps_per_epoch, batch_size))

    for perm in perms:
        batch = dataset[perm]
        batch = {k: np.array(v) for k, v in batch.items()}
        batch = flax_cu.shard(batch)

        yield batch


def glue_eval_data_collator(dataset: datasets.arrow_dataset.Dataset, batch_size: int):
    """Returns batches of size `batch_size` from `eval dataset`. Sharding handled by `pad_shard_unpad` in the eval loop."""
    batch_idx = np.arange(len(dataset))

    steps_per_epoch = math.ceil(len(dataset) / batch_size)
    batch_idx = np.array_split(batch_idx, steps_per_epoch)

    for idx in batch_idx:
        batch = dataset[idx]
        batch = {k: np.array(v) for k, v in batch.items()}

        yield batch
