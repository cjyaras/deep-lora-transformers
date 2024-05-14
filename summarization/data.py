import math
from typing import Any, Iterator, Optional, Tuple, cast

import datasets
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import transformers

import configs
import utils

# Summarization tasks
summarization_name_mapping = {
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


def load_dataset_from_config(
    task_config: configs.TaskConfig,
    pretrained_config: transformers.PretrainedConfig,
    seed: int,
) -> Tuple[
    datasets.arrow_dataset.Dataset,
    datasets.arrow_dataset.Dataset,
    datasets.arrow_dataset.Dataset,
]:
    tokenizer = transformers.AutoTokenizer.from_pretrained(task_config.pretrain_model)
    train_dataset, eval_dataset, predict_dataset = load_dataset(
        task_config.finetune_task_name,
        pretrained_config,
        tokenizer,
        task_config.max_source_length,
        task_config.max_target_length,
        task_config.num_train_samples,
        seed,
    )
    return train_dataset, eval_dataset, predict_dataset


def load_dataset(
    finetune_task_name: str,
    pretrained_config: transformers.PretrainedConfig,
    tokenizer: transformers.PreTrainedTokenizerBase,
    max_source_length: Optional[int],
    max_target_length: Optional[int],
    num_train_samples: Optional[int],
    sample_seed: int,
) -> Tuple[
    datasets.arrow_dataset.Dataset,
    datasets.arrow_dataset.Dataset,
    datasets.arrow_dataset.Dataset,
]:
    raw_datasets = datasets.load_dataset(finetune_task_name)
    text_column, summary_column = summarization_name_mapping[finetune_task_name]

    def length_of(example):
        text, summary = example[text_column], example[summary_column]
        return len(tokenizer(text)["input_ids"]), len(tokenizer(summary)["input_ids"])  # type: ignore

    if max_source_length is None:
        max_source_length = 1024

    if max_target_length is None:
        max_target_length = 128

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
        inputs = example[text_column]
        targets = example[summary_column]

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
        decoder_input_ids = utils.shift_tokens_right(
            labels["input_ids"],  # type: ignore
            pretrained_config.pad_token_id,
            pretrained_config.pad_token_id,
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
