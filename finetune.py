# Finetune a pretrained (Ro)BERT(a) on GLUE tasks
import math
from functools import partial

import evaluate
import flax
import flax.jax_utils as fju
import flax.training.common_utils as flax_cu
import jax
import numpy as np
import transformers
from flax.metrics.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import configs
import data
import train


def write_train_metric(summary_writer, train_metrics, step):
    train_metrics = flax_cu.get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)


def write_eval_metric(summary_writer, eval_metrics, step):
    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)


def finetune(
    model_args: configs.ModelArguments,
    data_args: configs.DataArguments,
    train_args: configs.TrainArguments,
):
    # Tokenizer
    print("Initializing tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.pretrain_model)

    # Dataset
    print("Initializing dataset...")
    train_dataset, eval_dataset, num_labels, is_regression = data.load_dataset(
        data_args.finetune_task,
        tokenizer,
        data_args.max_seq_length,
        data_args.max_train_samples,
        exclude_long_seq=True,
    )

    # Model
    config = transformers.AutoConfig.from_pretrained(
        model_args.pretrain_model,
        num_labels=num_labels,
        finetuning_task=data_args.finetune_task,
    )
    print("Initializing model...")
    model = transformers.FlaxAutoModelForSequenceClassification.from_pretrained(
        model_args.pretrain_model, config=config
    )

    train_batch_size = jax.local_device_count() * train_args.per_device_train_batch_size
    eval_batch_size = jax.local_device_count() * train_args.per_device_eval_batch_size

    assert train_batch_size <= len(train_dataset), "Batch size too large for dataset"

    learning_rate_fn = train.create_learning_rate_fn(
        train_ds_size=len(train_dataset),
        train_batch_size=train_batch_size,
        num_train_epochs=train_args.num_train_epochs,
        num_warmup_steps=train_args.warmup_steps,
        learning_rate=train_args.learning_rate,
        decay_ratio=train_args.decay_ratio,
    )

    use_lora = "lora" in train_args.finetune_strategy

    if not use_lora:

        @partial(jax.pmap, axis_name="batch", donate_argnums=(0,))
        def p_train_step(
            model_state: train.ModelTrainState,
            batch: dict[str, jax.Array],
            dropout_rng: jax.Array,
        ):
            dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
            targets = batch.pop("labels")

            def loss_fn(params):
                logits = model_state.apply_fn(
                    **batch, params=params, dropout_rng=dropout_rng, train=True
                )[0]
                loss = model_state.loss_fn(logits, targets)
                return loss

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grad = grad_fn(model_state.params)
            grad = jax.lax.pmean(grad, "batch")
            new_model_state = model_state.apply_gradients(grads=grad)
            metrics = jax.lax.pmean(
                {"loss": loss, "learning_rate": learning_rate_fn(model_state.step)},
                axis_name="batch",
            )
            return new_model_state, metrics, new_dropout_rng

        @partial(jax.pmap, axis_name="batch")
        def p_eval_step(
            model_state: train.ModelTrainState, batch: dict[str, jax.Array]
        ):
            logits = model_state.apply_fn(
                **batch, params=model_state.params, train=False
            )[0]
            return model_state.logits_fn(logits)

    else:

        @partial(jax.pmap, axis_name="batch", donate_argnums=(1,))
        def p_train_step(
            model_state: train.ModelTrainState,
            lora_state: train.LoraTrainState,
            batch: dict[str, jax.Array],
            dropout_rng: jax.Array,
        ):
            dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
            targets = batch.pop("labels")

            def loss_fn(lora_params):
                adapted_model_params = lora_state.apply_fn(
                    {"params": lora_params}, model.params
                )
                logits = model_state.apply_fn(
                    **batch,
                    params=adapted_model_params,
                    dropout_rng=dropout_rng,
                    train=True,
                )[0]
                loss = model_state.loss_fn(logits, targets)
                return loss

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grad = grad_fn(lora_state.params)
            grad = jax.lax.pmean(grad, "batch")
            new_lora_state = lora_state.apply_gradients(grads=grad)
            metrics = jax.lax.pmean(
                {"loss": loss, "learning_rate": learning_rate_fn(lora_state.step)},
                axis_name="batch",
            )
            return new_lora_state, metrics, new_dropout_rng

        @partial(jax.pmap, axis_name="batch")
        def p_eval_step(
            model_state: train.ModelTrainState,
            lora_state: train.LoraTrainState,
            batch: dict[str, jax.Array],
        ):
            adapted_model_params = lora_state.apply_fn(
                {"params": lora_state.params}, model.params
            )
            logits = model_state.apply_fn(
                **batch, params=adapted_model_params, train=False
            )[0]
            return model_state.logits_fn(logits)

    # Initialize model state
    model_state = train.create_model_train_state(
        model,
        learning_rate_fn,
        is_regression=is_regression,
        num_labels=num_labels,
        weight_decay=train_args.weight_decay,
        frozen=use_lora,
    )
    model_state = fju.replicate(model_state)
    lora_state = None
    if use_lora:
        lora_state = train.create_lora_train_state(
            model_args,
            model.params,
            depth=int(train_args.finetune_strategy[-1]),
            learning_rate_fn=learning_rate_fn,
            weight_decay=train_args.weight_decay,
        )
        lora_state = fju.replicate(lora_state)

    metric = evaluate.load("glue", data_args.finetune_task)

    assert (
        transformers.is_tensorboard_available()
    ), "Tensorboard is required for logging but is not installed."

    experiment_name = f"{model_args.pretrain_model}-{data_args.finetune_task}-{train_args.finetune_strategy}"
    if data_args.max_train_samples is not None:
        experiment_name += f"-{len(train_dataset)}"
    summary_writer = SummaryWriter(experiment_name)

    steps_per_epoch = len(train_dataset) // train_batch_size
    total_steps = steps_per_epoch * train_args.num_train_epochs
    epochs_pbar = tqdm(range(train_args.num_train_epochs))
    epochs_pbar.set_description("Epoch ")

    rng = jax.random.PRNGKey(train_args.seed)
    dropout_rngs = jax.random.split(rng, jax.local_device_count())

    if len(train_dataset) == train_batch_size:
        print(f"Full batch GD since len(dataset) = batch_size = {train_batch_size}")
        rng, input_rng = jax.random.split(rng)
        train_loader = data.glue_train_data_collator(
            input_rng, train_dataset, train_batch_size  # type: ignore
        )
        train_batch = next(train_loader)
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
            if cur_step % train_args.log_steps == 0 or cur_step % total_steps == 0:
                # Save metrics
                train_metric = flax.jax_utils.unreplicate(train_metric)
                write_train_metric(summary_writer, train_metrics, cur_step)

                train_metrics = []

            if cur_step % train_args.eval_steps == 0 or cur_step % total_steps == 0:
                # evaluate
                eval_loader_pbar = tqdm(data.glue_eval_data_collator(eval_dataset, eval_batch_size), leave=False, total=math.ceil(len(eval_dataset) / eval_batch_size))  # type: ignore
                eval_loader_pbar.set_description(f"Evaluating ")
                for eval_batch in eval_loader_pbar:
                    labels = eval_batch.pop("labels")  # type: ignore
                    if not use_lora:
                        predictions = flax.jax_utils.pad_shard_unpad(p_eval_step)(
                            model_state,
                            eval_batch,
                            min_device_batch=train_args.per_device_eval_batch_size,
                        )
                    else:
                        predictions = flax.jax_utils.pad_shard_unpad(
                            p_eval_step, static_argnums=(0, 1)
                        )(
                            model_state,
                            lora_state,
                            eval_batch,
                            min_device_batch=train_args.per_device_eval_batch_size,
                        )
                    metric.add_batch(
                        predictions=np.array(predictions), references=labels
                    )

                eval_metric = metric.compute()

                print(f"Step ({cur_step}/{total_steps} | Eval metrics: {eval_metric})")

                write_eval_metric(summary_writer, eval_metric, cur_step)
    else:
        print(
            f"SGD since {len(train_dataset)} = len(dataset) != batch_size = {train_batch_size}"
        )
        for epoch in epochs_pbar:
            train_metrics = []
            rng, input_rng = jax.random.split(rng)
            train_loader = data.glue_train_data_collator(
                input_rng, train_dataset, train_batch_size  # type: ignore
            )
            for step, train_batch in enumerate(train_loader):
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

                if cur_step % train_args.log_steps == 0 or cur_step % total_steps == 0:
                    # Save metrics
                    train_metric = flax.jax_utils.unreplicate(train_metric)
                    write_train_metric(summary_writer, train_metrics, cur_step)

                    train_metrics = []

                if cur_step % train_args.eval_steps == 0 or cur_step % total_steps == 0:
                    # evaluate
                    eval_loader_pbar = tqdm(data.glue_eval_data_collator(eval_dataset, eval_batch_size), leave=False, total=math.ceil(len(eval_dataset) / eval_batch_size))  # type: ignore
                    eval_loader_pbar.set_description(f"Evaluating ")
                    for batch in eval_loader_pbar:
                        labels = batch.pop("labels")  # type: ignore
                        if not use_lora:
                            predictions = flax.jax_utils.pad_shard_unpad(p_eval_step)(
                                model_state,
                                batch,
                                min_device_batch=train_args.per_device_eval_batch_size,
                            )
                        else:
                            predictions = flax.jax_utils.pad_shard_unpad(
                                p_eval_step, static_argnums=(0, 1)
                            )(
                                model_state,
                                lora_state,
                                batch,
                                min_device_batch=train_args.per_device_eval_batch_size,
                            )
                        metric.add_batch(
                            predictions=np.array(predictions), references=labels
                        )

                    eval_metric = metric.compute()

                    print(
                        f"Step ({cur_step}/{total_steps} | Eval metrics: {eval_metric})"
                    )

                    write_eval_metric(summary_writer, eval_metric, cur_step)

    return model_state, lora_state


def main():
    model_args = configs.ModelArguments()
    data_args = configs.DataArguments()
    train_args = configs.TrainArguments()

    finetune(model_args, data_args, train_args)


if __name__ == "__main__":
    main()
