import math
from typing import Any, Iterator, Optional, Tuple, cast

import configs
import datasets
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import transformers

# Glue tasks
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
}

task_to_num_labels = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "qnli": 2,
    "qqp": 2,
    "rte": 2,
    "sst2": 2,
    "stsb": 1,
}


def load_dataset_from_config(
    task_config: configs.TaskConfig, seed: int
) -> Tuple[datasets.arrow_dataset.Dataset, datasets.arrow_dataset.Dataset]:
    tokenizer = transformers.AutoTokenizer.from_pretrained(task_config.pretrain_model)
    train_dataset, eval_dataset = load_dataset(
        task_config.finetune_task_name,
        tokenizer,
        task_config.max_seq_length,
        task_config.num_train_samples,
        seed,
    )
    return train_dataset, eval_dataset


def load_dataset(
    finetune_task_name: str,
    tokenizer: transformers.PreTrainedTokenizerBase,
    max_seq_length: Optional[int],
    num_train_samples: Optional[int],
    sample_seed: int,
) -> Tuple[datasets.arrow_dataset.Dataset, datasets.arrow_dataset.Dataset]:
    raw_datasets = datasets.load_dataset("glue", finetune_task_name)

    def length_of(example):
        texts = (
            (example[sentence1_key],)
            if sentence2_key is None
            else (example[sentence1_key], example[sentence2_key])
        )
        return len(tokenizer(*texts)["input_ids"])  # type: ignore

    # Preprocess dataset
    sentence1_key, sentence2_key = task_to_keys[finetune_task_name]

    if max_seq_length is not None:
        for k, v in raw_datasets.items():  # type: ignore
            raw_datasets[k] = v.filter(lambda example: length_of(example) <= max_seq_length)  # type: ignore

    if num_train_samples is not None:
        raw_datasets["train"] = raw_datasets["train"].select(  # type: ignore
            jr.choice(
                jax.random.PRNGKey(sample_seed),
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
    train_dataset = cast(datasets.arrow_dataset.Dataset, train_dataset)
    eval_dataset = cast(datasets.arrow_dataset.Dataset, eval_dataset)
    return train_dataset, eval_dataset


def create_train_iterator(
    rng: jax.Array, dataset: datasets.arrow_dataset.Dataset, batch_size: int
) -> Iterator[dict[Any, np.ndarray]]:
    assert batch_size <= len(dataset), "Batch size must be smaller than dataset size."
    if len(dataset) == batch_size:
        # If dataset size is equal to batch size, then we can just return the dataset as a batch.
        batch = dataset[:]
        batch = {k: np.array(v) for k, v in batch.items()}

        while True:
            yield batch
    else:
        steps_per_epoch = len(dataset) // batch_size
        while True:
            perm_rng, rng = jr.split(rng)
            perms = jr.permutation(perm_rng, len(dataset))
            perms = perms[: steps_per_epoch * batch_size]  # Skip incomplete batch.
            perms = perms.reshape((steps_per_epoch, batch_size))

            for perm in perms:
                batch = dataset[perm]
                batch = {k: np.array(v) for k, v in batch.items()}

                yield batch


def create_eval_iterator(
    dataset: datasets.arrow_dataset.Dataset, batch_size: int
) -> Iterator[dict[Any, np.ndarray]]:
    batch_idx = np.arange(len(dataset))

    steps_per_epoch = math.ceil(len(dataset) / batch_size)
    batch_idx = np.array_split(batch_idx, steps_per_epoch)

    for idx in batch_idx:
        batch = dataset[idx]
        batch = {k: np.array(v) for k, v in batch.items()}

        yield batch
