import math
from typing import Any, Iterator, Optional, Tuple, cast

import datasets
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import transformers
from datasets.arrow_dataset import Dataset

import data_utils
from configs import GlueTaskName, SummarizationTaskName, TaskConfig, TaskType

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

# Summarization tasks
SUMMARIZATION_TASK_TO_KEYS = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}

BART_PAD_TOKEN_ID = 1
BART_DECODER_START_TOKEN_ID = 2


def load_dataset_from_config(task_config: TaskConfig, sample_seed: int) -> Tuple[
    Dataset,
    Dataset,
    Optional[Dataset],
]:
    tokenizer = transformers.AutoTokenizer.from_pretrained(task_config.pretrain_model)

    if task_config.task_type == TaskType.GLUE:
        assert isinstance(task_config.finetune_task_name, GlueTaskName)
        assert isinstance(task_config.max_seq_length, int)
        train_dataset, eval_dataset = load_glue_dataset(
            task_config.finetune_task_name,
            tokenizer,
            task_config.max_seq_length,
            task_config.num_train_samples,
            sample_seed,
        )
        predict_dataset = None
    elif task_config.task_type == TaskType.SUMMARIZATION:
        assert isinstance(task_config.finetune_task_name, SummarizationTaskName)
        assert isinstance(task_config.max_seq_length, Tuple)
        train_dataset, eval_dataset, predict_dataset = load_summarization_dataset(
            task_config.finetune_task_name,
            tokenizer,
            task_config.max_seq_length,
            task_config.num_train_samples,
            sample_seed,
        )
    else:
        raise ValueError(f"Task type {task_config.task_type} not supported.")
    return train_dataset, eval_dataset, predict_dataset


def load_glue_dataset(
    finetune_task_name: GlueTaskName,
    tokenizer: transformers.PreTrainedTokenizerBase,
    max_seq_length: Optional[int],
    num_train_samples: Optional[int],
    sample_seed: int,
) -> Tuple[Dataset, Dataset]:

    raw_datasets = datasets.load_dataset("glue", finetune_task_name)

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


def load_summarization_dataset(
    finetune_task_name: SummarizationTaskName,
    tokenizer: transformers.PreTrainedTokenizerBase,
    max_seq_length: Optional[Tuple[int, int]],
    num_train_samples: Optional[int],
    sample_seed: int,
) -> Tuple[Dataset, Dataset, Dataset]:

    raw_datasets = datasets.load_dataset(finetune_task_name)

    text_key, summary_key = SUMMARIZATION_TASK_TO_KEYS[finetune_task_name]

    def length_of(example):
        text, summary = example[text_key], example[summary_key]
        return len(tokenizer(text)["input_ids"]), len(tokenizer(summary)["input_ids"])  # type: ignore

    if max_seq_length is None:
        max_source_length = 1024
        max_target_length = 128
    else:
        max_source_length, max_target_length = max_seq_length

    def keep_fn(example):
        text_length, summary_length = length_of(example)
        return text_length <= max_source_length and summary_length <= max_target_length

    for k, v in raw_datasets.items():  # type: ignore
        raw_datasets[k] = v.filter(keep_fn)  # type: ignore

    if num_train_samples is not None:
        raw_datasets["train"] = raw_datasets["train"].select(  # type: ignore
            jr.choice(
                jax.random.key(sample_seed),
                jnp.arange(len(raw_datasets["train"])),  # type: ignore
                shape=(num_train_samples,),
            )
        )

    def preprocess_fn(example):
        inputs = example[text_key]
        targets = example[summary_key]

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
            BART_PAD_TOKEN_ID,
            BART_DECODER_START_TOKEN_ID,
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
    predict_dataset = cast(
        datasets.arrow_dataset.Dataset,
        processed_datasets["test"],  # type: ignore
    )

    return train_dataset, eval_dataset, predict_dataset


def create_train_iterator(
    rng: jax.Array, dataset: datasets.arrow_dataset.Dataset, batch_size: int
) -> Iterator[dict[Any, np.ndarray]]:
    assert batch_size <= len(dataset), "Batch size must be smaller than dataset size."
    if len(dataset) == batch_size:
        # If dataset size is equal to batch size, then we can just return the dataset as a batch.
        batch = dataset[:]

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

                yield batch


def create_eval_iterator(
    dataset: datasets.arrow_dataset.Dataset, batch_size: int
) -> Iterator[dict[Any, np.ndarray]]:
    batch_idx = np.arange(len(dataset))

    steps_per_epoch = math.ceil(len(dataset) / batch_size)
    batch_idx = np.array_split(batch_idx, steps_per_epoch)

    for idx in batch_idx:
        batch = dataset[idx]

        yield batch
