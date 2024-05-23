import json
import os

import numpy as np

# from dlt import configs, metrics
# from dlt.finetune import finetune


# def common_config(num_train_steps):
#     task_config = configs.TaskConfig()
#     task_config.task_type = configs.TaskType.NLG
#     task_config.pretrain_model = configs.ModelType.T5
#     task_config.num_train_samples = 16
#     task_config.train_batch_size = 16
#     task_config.eval_batch_size = 32
#     task_config.max_seq_length = (64, 64)
#     task_config.lora_init_scale = 1e-3
#     task_config.num_train_steps = num_train_steps
#     task_config.log_eval_steps = num_train_steps
#     # task_config.log_eval_steps = 10
#     task_config.decay_ratio = 1.0
#     task_config.lora_adapt_type = configs.LoraAdaptType.ALL_DENSE
#     return task_config


# def run_experiments(
#     finetune_task_name, depth, rank, learning_rate, seeds, compress=False
# ):
#     task_config = common_config(400)
#     task_config.finetune_task_name = finetune_task_name
#     task_config.save_dir = f"../checkpoints/nlg_fewshot/{finetune_task_name}"
#     task_config.lora_depth = depth
#     task_config.lora_rank = rank
#     task_config.learning_rate = learning_rate
#     if compress:
#         task_config.lora_compress = True
#         task_config.lora_gamma = 1e-2

#     finetune(task_config, seeds)


# def main():
#     seeds = np.arange(10)
#     for finetune_task_name in configs.NLGTaskName.values():
#         run_experiments(
#             finetune_task_name,
#             depth=2,
#             rank=8,
#             learning_rate=1e-4,
#             seeds=seeds,
#         )
#         run_experiments(
#             finetune_task_name,
#             depth=3,
#             rank=8,
#             learning_rate=1e-4,
#             seeds=seeds,
#             compress=False,
#         )


NLG_METRIC_LIST = [
    "eval_bleu",
    "eval_rouge1",
    "eval_rouge2",
    "eval_rougeL",
    "eval_rougeLsum",
    "eval_meteor",
]


def get_results():
    experiment_dir = "../checkpoints/nlg_fewshot"
    tasks = os.listdir(experiment_dir)

    diff_dict = {}
    take_last = True

    def get_seed(run):
        return int(run.split("_")[-1].split("=")[1])

    for task in sorted(tasks):
        runs = [run for run in os.listdir(os.path.join(experiment_dir, task))]
        runs_2 = [run for run in runs if "depth=2" in run]
        runs_3 = [run for run in runs if "depth=3" in run]
        runs_2_results = np.zeros(len(runs_2))
        runs_3_results = np.zeros(len(runs_3))
        for run in runs:
            seed = get_seed(run)
            with open(os.path.join(experiment_dir, task, run, "results.json")) as f:
                results = json.load(f)
            values = [
                entry["value"]
                for entry in sorted(
                    results[NLG_METRIC_LIST[5]], key=lambda x: x["step"]
                )
            ]
            if "depth=2" in run:
                runs_2_results[seed] = values[-1] if take_last else max(values)
            elif "depth=3" in run:
                runs_3_results[seed] = values[-1] if take_last else max(values)
        diff_dict[task] = runs_3_results - runs_2_results

    labels, series = list(zip(*diff_dict.items()))
    labels = list(labels)
    series = list(series)
    series.append(np.array(series).reshape(-1))
    labels.append("overall")
    return series, labels


if __name__ == "__main__":
    # main()

    results, labels = get_results()
    for series, tag in zip(results, labels):
        print(tag, np.mean(series), np.var(series))
