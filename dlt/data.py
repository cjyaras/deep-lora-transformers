import math
from typing import Any, Iterator, Optional, Tuple, cast

import datasets
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import transformers
from datasets.arrow_dataset import Dataset
from transformers import PretrainedConfig, T5Config

from . import data_utils
from .configs import GlueTaskName, NLGTaskName, TaskConfig, TaskType

# Glue tasks
GLUE_TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
}


def load_dataset_from_config(
    task_config: TaskConfig, model_config: PretrainedConfig, sample_seed: int
) -> Tuple[Dataset, Dataset]:
    tokenizer = transformers.AutoTokenizer.from_pretrained(task_config.pretrain_model)

    if task_config.task_type == TaskType.GLUE:
        assert task_config.finetune_task_name in GlueTaskName.values()
        assert isinstance(task_config.max_seq_length, int)
        train_dataset, eval_dataset = load_glue_dataset(
            task_config.finetune_task_name,  # type: ignore
            tokenizer,
            task_config.max_seq_length,
            task_config.num_train_samples,
            sample_seed,
        )
    elif task_config.task_type == TaskType.NLG:
        assert task_config.finetune_task_name in NLGTaskName.values()
        assert isinstance(task_config.max_seq_length, Tuple)
        train_dataset, eval_dataset = load_e2e_nlg_dataset(
            model_config,
            tokenizer,
            task_config.max_seq_length,
            task_config.num_train_samples,
            sample_seed,
        )
    else:
        raise ValueError(f"Task type {task_config.task_type} not supported.")
    return train_dataset, eval_dataset


def load_glue_dataset(
    finetune_task_name: GlueTaskName,
    tokenizer: transformers.PreTrainedTokenizerBase,
    max_seq_length: Optional[int],
    num_train_samples: Optional[int],
    sample_seed: int,
) -> Tuple[Dataset, Dataset]:

    raw_datasets = datasets.load_dataset("nyu-mll/glue", finetune_task_name)

    def length_of(example):
        texts = (
            (example[sentence1_key],)
            if sentence2_key is None
            else (example[sentence1_key], example[sentence2_key])
        )
        return len(tokenizer(*texts)["input_ids"])  # type: ignore

    # Preprocess dataset
    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[finetune_task_name]

    if max_seq_length is not None:
        for k, v in raw_datasets.items():  # type: ignore
            raw_datasets[k] = v.filter(lambda example: length_of(example) <= max_seq_length)  # type: ignore

    if num_train_samples is not None:
        raw_datasets["train"] = raw_datasets["train"].select(  # type: ignore
            jr.choice(
                jax.random.key(sample_seed),
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
            *texts,
            padding="max_length",
            max_length=max_seq_length,
            truncation=True,
            return_tensors="np",
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


def load_e2e_nlg_dataset(
    model_config: PretrainedConfig,
    tokenizer: transformers.PreTrainedTokenizerBase,
    max_seq_length: Optional[Tuple[int, int]],
    num_train_samples: Optional[int],
    sample_seed: int,
) -> Tuple[Dataset, Dataset]:

    raw_datasets = datasets.load_dataset("GEM/e2e_nlg")
    source_key = "meaning_representation"
    target_key = "target"

    def length_of(example):
        source, target = example[source_key], example[target_key]
        return len(tokenizer(source)["input_ids"]), len(tokenizer(target)["input_ids"])  # type: ignore

    if max_seq_length is None:
        max_source_length = 64
        max_target_length = 64
    else:
        max_source_length, max_target_length = max_seq_length

    def keep_fn(example):
        source_length, target_length = length_of(example)
        return source_length <= max_source_length and target_length <= max_target_length

    for k, v in raw_datasets.items():  # type: ignore
        raw_datasets[k] = v.filter(keep_fn, num_proc=10)  # type: ignore

    if num_train_samples is not None:
        raw_datasets["train"] = raw_datasets["train"].select(  # type: ignore
            jr.choice(
                jax.random.key(sample_seed),
                jnp.arange(len(raw_datasets["train"])),  # type: ignore
                shape=(num_train_samples,),
            )
        )

    def preprocess_fn(example):
        inputs = example[source_key]
        targets = example[target_key]

        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )

        labels = tokenizer(
            targets,
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        model_inputs["labels"] = labels["input_ids"]
        decoder_input_ids = data_utils.shift_tokens_right(
            labels["input_ids"],  # type: ignore
            model_config.decoder_start_token_id,
        )
        model_inputs["decoder_input_ids"] = decoder_input_ids
        model_inputs["decoder_attention_mask"] = labels["attention_mask"]

        return model_inputs

    processed_datasets = raw_datasets.map(
        preprocess_fn,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,  # type: ignore
    )

    train_dataset = cast(
        datasets.arrow_dataset.Dataset,
        processed_datasets["train"],  # type: ignore
    )
    eval_dataset = cast(
        datasets.arrow_dataset.Dataset,
        processed_datasets["validation"],  # type: ignore
    )

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
