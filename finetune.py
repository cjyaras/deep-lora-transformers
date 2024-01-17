# Finetune a pretrained (Ro)BERT(a) on GLUE tasks
import math

import evaluate
import flax
import flax.jax_utils as fju
import jax
import numpy as np
import transformers
from flax.metrics.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import configs
import data
import models
import train
import utils


def finetune(task_config: configs.TaskConfig):
    (
        train_dataset,
        eval_dataset,
        num_labels,
        is_regression,
    ) = data.load_dataset_from_config(task_config)

    # Model
    pretrain_model = models.create_pretrain_model_from_config(task_config, num_labels)

    train_batch_size = (
        jax.local_device_count() * task_config.per_device_train_batch_size
    )
    eval_batch_size = jax.local_device_count() * task_config.per_device_eval_batch_size

    assert train_batch_size <= len(train_dataset), "Batch size too large for dataset"

    learning_rate_fn = train.create_learning_rate_fn(
        train_ds_size=len(train_dataset),
        train_batch_size=train_batch_size,
        num_train_epochs=task_config.num_train_epochs,
        num_warmup_steps=task_config.warmup_steps,
        learning_rate=task_config.learning_rate,
        decay_ratio=task_config.decay_ratio,
    )

    use_lora = task_config.finetune_strategy == "lora"

    p_train_step, p_eval_step = train.create_train_eval_step_fns(
        learning_rate_fn, use_lora
    )

    # Initialize model state
    model_state = train.create_model_train_state(
        pretrain_model,
        learning_rate_fn,
        is_regression=is_regression,
        num_labels=num_labels,
        weight_decay=task_config.weight_decay,
        frozen=use_lora,
    )
    model_state = fju.replicate(model_state)
    lora_state = None
    rng = jax.random.PRNGKey(task_config.train_seed)
    if use_lora:
        lora_rng, rng = jax.random.split(rng)
        # filter_fn = lambda _, v: len(v) == 2 and min(v) > 100
        filter_fn = lambda k, _: "query/kernel" in k or "key/kernel" in k
        lora_state = train.create_lora_train_state(
            task_config,
            pretrain_model.params,
            learning_rate_fn=learning_rate_fn,
            lora_rng=lora_rng,
        )
        lora_state = fju.replicate(lora_state)

    metric = evaluate.load("glue", task_config.finetune_task_name)

    assert (
        transformers.is_tensorboard_available()
    ), "Tensorboard is required for logging but is not installed."

    experiment_name = (
        f"{task_config.finetune_task_name}_{task_config.finetune_strategy}"
    )
    if use_lora:
        experiment_name += f"_{task_config.lora_depth}-{task_config.lora_rank}"
    if task_config.num_train_samples is not None:
        experiment_name += f"_samples={len(train_dataset)}"
    summary_writer = SummaryWriter(experiment_name)

    steps_per_epoch = len(train_dataset) // train_batch_size
    total_steps = steps_per_epoch * task_config.num_train_epochs

    all_rngs = jax.random.split(rng, jax.local_device_count() + 1)
    dropout_rngs = all_rngs[:-1]
    rng = all_rngs[-1]

    if len(train_dataset) == train_batch_size:
        print(f"Full batch GD since len(dataset) = batch_size = {train_batch_size}")
        rng, input_rng = jax.random.split(rng)
        train_loader = data.glue_train_data_collator(
            input_rng, train_dataset, train_batch_size  # type: ignore
        )
        train_batch = next(train_loader)
        epochs_pbar = tqdm(range(task_config.num_train_epochs))
        epochs_pbar.set_description("Epoch ")
        for epoch in epochs_pbar:
            train_metrics = []

            if not use_lora:
                model_state, train_metric, dropout_rngs = p_train_step(
                    model_state, train_batch.copy(), dropout_rngs
                )
            else:
                lora_state, train_metric, dropout_rngs = p_train_step(
                    model_state, lora_state, train_batch.copy(), dropout_rngs
                )
            train_metrics.append(train_metric)

            cur_step = epoch + 1
            if cur_step % task_config.log_steps == 0 or cur_step % total_steps == 0:
                # Save metrics
                train_metric = flax.jax_utils.unreplicate(train_metric)
                utils.write_train_metric(summary_writer, train_metrics, cur_step)

                train_metrics = []

            if cur_step % task_config.eval_steps == 0 or cur_step % total_steps == 0:
                # evaluate
                eval_loader_pbar = tqdm(data.glue_eval_data_collator(eval_dataset, eval_batch_size), leave=False, total=math.ceil(len(eval_dataset) / eval_batch_size))  # type: ignore
                eval_loader_pbar.set_description(f"Evaluating ")
                for eval_batch in eval_loader_pbar:
                    labels = eval_batch.pop("labels")  # type: ignore
                    if not use_lora:
                        predictions = flax.jax_utils.pad_shard_unpad(p_eval_step)(
                            model_state,
                            eval_batch,
                            min_device_batch=task_config.per_device_eval_batch_size,
                        )
                    else:
                        predictions = flax.jax_utils.pad_shard_unpad(
                            p_eval_step, static_argnums=(0, 1)
                        )(
                            model_state,
                            lora_state,
                            eval_batch,
                            min_device_batch=task_config.per_device_eval_batch_size,
                        )
                    metric.add_batch(
                        predictions=np.array(predictions), references=labels
                    )

                eval_metric = metric.compute()

                print(f"Step ({cur_step}/{total_steps} | Eval metrics: {eval_metric})")

                utils.write_eval_metric(summary_writer, eval_metric, cur_step)
    else:
        print(
            f"SGD since {len(train_dataset)} = len(dataset) != batch_size = {train_batch_size}"
        )
        epochs_pbar = tqdm(range(task_config.num_train_epochs))
        epochs_pbar.set_description("Epoch ")
        for epoch in epochs_pbar:
            train_metrics = []
            rng, input_rng = jax.random.split(rng)
            train_loader_pbar = tqdm(
                data.glue_train_data_collator(
                    input_rng, train_dataset, train_batch_size  # type: ignore
                ),
                leave=False,
                total=math.ceil(len(train_dataset) / train_batch_size),
            )
            for step, train_batch in enumerate(train_loader_pbar):
                if not use_lora:
                    model_state, train_metric, dropout_rngs = p_train_step(
                        model_state, train_batch, dropout_rngs
                    )
                else:
                    lora_state, train_metric, dropout_rngs = p_train_step(
                        model_state, lora_state, train_batch, dropout_rngs
                    )
                train_metrics.append(train_metric)

                cur_step = (epoch * steps_per_epoch) + (step + 1)

                if cur_step % task_config.log_steps == 0 or cur_step % total_steps == 0:
                    # Save metrics
                    train_metric = flax.jax_utils.unreplicate(train_metric)
                    utils.write_train_metric(summary_writer, train_metrics, cur_step)

                    train_metrics = []

                if (
                    cur_step % task_config.eval_steps == 0
                    or cur_step % total_steps == 0
                ):
                    # evaluate
                    eval_loader_pbar = tqdm(data.glue_eval_data_collator(eval_dataset, eval_batch_size), leave=False, total=math.ceil(len(eval_dataset) / eval_batch_size))  # type: ignore
                    eval_loader_pbar.set_description(f"Evaluating ")
                    for batch in eval_loader_pbar:
                        labels = batch.pop("labels")  # type: ignore
                        if not use_lora:
                            predictions = flax.jax_utils.pad_shard_unpad(p_eval_step)(
                                model_state,
                                batch,
                                min_device_batch=task_config.per_device_eval_batch_size,
                            )
                        else:
                            predictions = flax.jax_utils.pad_shard_unpad(
                                p_eval_step, static_argnums=(0, 1)
                            )(
                                model_state,
                                lora_state,
                                batch,
                                min_device_batch=task_config.per_device_eval_batch_size,
                            )
                        metric.add_batch(
                            predictions=np.array(predictions), references=labels
                        )

                    eval_metric = metric.compute()

                    print(
                        f"Step ({cur_step}/{total_steps} | Eval metrics: {eval_metric})"
                    )

                    utils.write_eval_metric(summary_writer, eval_metric, cur_step)

    return model_state, lora_state


def main():
    task_config = configs.TaskConfig()
    task_config.finetune_strategy = "lora"
    task_config.lora_rank = 4
    print(task_config)
    finetune(task_config)


if __name__ == "__main__":
    main()
