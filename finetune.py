import math
import os
import shutil

import evaluate
import flax
import flax.metrics.tensorboard
import jax
import tqdm.auto as tqdm_lib
import transformers

import configs
import data
import models
import train
import utils

experiment_dir = os.path.join(os.getcwd(), "experiments")


def finetune(task_config: configs.TaskConfig):
    (
        train_dataset,
        eval_dataset,
        num_labels,
        is_regression,
    ) = data.load_dataset_from_config(task_config)

    # Model
    pretrain_model = models.create_pretrain_model_from_config(task_config, num_labels)

    learning_rate_fn = train.create_learning_rate_fn(
        task_config.num_train_steps,
        task_config.num_warmup_steps,
        task_config.learning_rate,
        task_config.decay_ratio,
    )

    train_step, eval_step = train.create_train_eval_step_fns(learning_rate_fn)

    model_state = train.create_model_state(
        model=pretrain_model,
        is_regression=is_regression,
    )

    rng = jax.random.PRNGKey(task_config.train_seed)

    lora_rng, rng = jax.random.split(rng)
    uncompressed_lora_state = train.create_lora_train_state(
        task_config,
        pretrain_model.params,  # type: ignore
        learning_rate_fn=learning_rate_fn,
        lora_rng=lora_rng,
    )

    eval_metric = evaluate.load("glue", task_config.finetune_task_name)

    assert (
        transformers.is_tensorboard_available()
    ), "Tensorboard is required for logging but is not installed."

    experiment_name = utils.get_experiment_name(task_config)
    experiment_path = os.path.join(experiment_dir, experiment_name)

    if os.path.exists(experiment_path):
        shutil.rmtree(experiment_path)
    summary_writer = flax.metrics.tensorboard.SummaryWriter(experiment_path)

    with open(os.path.join(experiment_path, "config.json"), "w") as f:
        f.write(task_config.to_json(indent=4))  # type: ignore

    dropout_rng, input_rng, rng = jax.random.split(rng, 3)
    train_iterator = data.create_train_iterator(
        input_rng, train_dataset, task_config.train_batch_size
    )

    # Compression
    if task_config.lora_compress:
        batch = next(train_iterator)
        lora_state = train.create_compressed_lora_train_state(
            uncompressed_lora_state, model_state, batch, dropout_rng, task_config
        )
        del uncompressed_lora_state
        train_iterator = data.create_train_iterator(
            input_rng, train_dataset, task_config.train_batch_size
        )
        print(flax.traverse_util.flatten_dict(lora_state.params, sep="/").keys())
    else:
        lora_state = uncompressed_lora_state

    train_metrics = []
    tqdm_train_iterator = tqdm_lib.tqdm(
        train_iterator, total=task_config.num_train_steps
    )

    if 0 in task_config.save_step_points:
        utils.save_lora_params(experiment_path, 0, lora_state.params)

    for step, train_batch in enumerate(tqdm_train_iterator, 1):
        # Iterator is infinite, need to break out
        if step > task_config.num_train_steps:
            break

        lora_state, train_metric, dropout_rng = train_step(
            model_state, lora_state, train_batch, dropout_rng
        )
        train_metrics.append(train_metric)

        if (
            step % task_config.log_eval_steps == 0
            or step == task_config.num_train_steps
        ):
            utils.write_train_metric(summary_writer, train_metrics, step)
            train_metrics = []

            eval_iterator = data.create_eval_iterator(
                eval_dataset, task_config.eval_batch_size
            )
            tqdm_eval_iterator = tqdm_lib.tqdm(
                eval_iterator,
                leave=False,
                total=math.ceil(len(eval_dataset) / task_config.eval_batch_size),
            )
            for eval_batch in tqdm_eval_iterator:
                padded_eval_batch, labels = utils.pad_to_batch_size(
                    eval_batch, task_config.eval_batch_size
                )
                padded_predictions = eval_step(
                    model_state, lora_state, padded_eval_batch
                )
                predictions = padded_predictions[: len(labels)]
                eval_metric.add_batch(predictions=predictions, references=labels)

            eval_metric_value = eval_metric.compute()
            utils.write_eval_metric(summary_writer, eval_metric_value, step)
            print(
                f"Step ({step}/{task_config.num_train_steps}), Eval Metrics: {eval_metric_value}"
            )

        if step in task_config.save_step_points:
            utils.save_lora_params(experiment_path, step, lora_state.params)
