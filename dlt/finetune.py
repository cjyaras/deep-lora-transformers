import json
import math
import os

import flax
import flax.metrics.tensorboard
import jax
import tqdm.auto as tqdm_lib
import transformers

from . import configs, data, data_utils, logging_utils, metrics, models, train


def finetune(task_config: configs.TaskConfig, seeds: list[int] = [0]):
    assert len(seeds) >= 1, "Need at least one seed"

    # Model
    model_config = models.create_pretrain_config_from_config(task_config)
    pretrain_model = models.create_pretrain_model_from_config(
        task_config.pretrain_model, model_config
    )

    learning_rate_fn = train.create_learning_rate_fn(
        task_config.num_train_steps,
        task_config.num_warmup_steps,
        task_config.learning_rate,
        task_config.decay_ratio,
    )

    train_step = train.create_train_step_fn(task_config, learning_rate_fn)

    if task_config.task_type == configs.TaskType.GLUE:
        assert isinstance(task_config.finetune_task_name, configs.GlueTaskName)
        eval_metric = metrics.GlueEvalMetric(task_config.finetune_task_name)
        eval_step = train.create_eval_step_fn(task_config)
    elif task_config.task_type == configs.TaskType.E2E_NLG:
        eval_metric = metrics.NLGEvalMetric(task_config.pretrain_model)
        eval_step = train.create_decode_step_fn(pretrain_model, task_config)

    assert (
        transformers.is_tensorboard_available()
    ), "Tensorboard is required for logging but is not installed."

    for seed in seeds:
        train_dataset, eval_dataset = data.load_dataset_from_config(
            task_config, model_config, seed
        )
        rng = jax.random.PRNGKey(seed)
        lora_rng, rng = jax.random.split(rng)

        experiment_path = logging_utils.get_experiment_path(task_config, seed)
        print(experiment_path)

        if os.path.exists(experiment_path):
            print(f"Experiment {experiment_path} already exists.")
            continue

        summary_writer = flax.metrics.tensorboard.SummaryWriter(experiment_path)

        with open(os.path.join(experiment_path, "config.json"), "w") as f:
            f.write(task_config.to_json(indent=4))  # type: ignore

        dropout_rng, input_rng, rng = jax.random.split(rng, 3)
        train_iterator = data.create_train_iterator(
            input_rng, train_dataset, task_config.train_batch_size
        )

        model_state = train.create_model_state(model=pretrain_model)

        (
            uncompressed_lora_state,
            uncompressed_lora_model,
        ) = train.create_lora_state(
            task_config,
            pretrain_model.params,  # type: ignore
            learning_rate_fn=learning_rate_fn,
            lora_rng=lora_rng,
            dropout_rng=dropout_rng,
        )

        # Compression
        if task_config.lora_compress:
            batch = next(train_iterator)
            lora_state = train.create_compressed_lora_train_state(
                uncompressed_lora_state,
                uncompressed_lora_model,
                model_state,
                batch,
                task_config,
            )
            del uncompressed_lora_state
            train_iterator = data.create_train_iterator(
                input_rng, train_dataset, task_config.train_batch_size
            )
        else:
            lora_state = uncompressed_lora_state

        print(
            "Number of trainable parameters: ",
            sum(
                jax.tree_util.tree_map(
                    jax.numpy.size, jax.tree_util.tree_leaves(lora_state.params)
                )
            ),
        )
        train_metrics = []
        tqdm_train_iterator = tqdm_lib.tqdm(
            train_iterator, total=task_config.num_train_steps
        )

        if 0 in task_config.save_step_points:
            logging_utils.save_lora_params(experiment_path, 0, lora_state.params)

        result_dict = {}

        for step, train_batch in enumerate(tqdm_train_iterator, 1):
            if step > task_config.num_train_steps:
                break

            lora_state, train_metric = train_step(model_state, lora_state, train_batch)
            train_metrics.append(train_metric)

            if (
                step % task_config.log_eval_steps == 0
                or step == task_config.num_train_steps
            ):
                logging_utils.write_train_metric(
                    summary_writer, result_dict, train_metrics, step
                )
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
                    padded_eval_batch, labels = data_utils.pad_to_batch_size(
                        eval_batch, task_config.eval_batch_size
                    )
                    padded_predictions = eval_step(
                        model_state, lora_state, padded_eval_batch
                    )
                    predictions = padded_predictions[: len(labels)]
                    eval_metric.add_batch(predictions=predictions, references=labels)

                eval_metric_value = eval_metric.compute()
                logging_utils.write_eval_metric(
                    summary_writer, result_dict, eval_metric_value, step
                )
                print(
                    f"Step ({step}/{task_config.num_train_steps}), Eval Metrics: {eval_metric_value}"
                )

            if step in task_config.save_step_points:
                logging_utils.save_lora_params(experiment_path, step, lora_state.params)

        with open(os.path.join(experiment_path, "results.json"), "w") as f:
            json.dump(result_dict, f)
