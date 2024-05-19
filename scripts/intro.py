import os
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from dlt import configs, logging_utils, models, plot_utils
from dlt.finetune import finetune

task_config = configs.TaskConfig()

task_config.task_type = configs.TaskType.GLUE
task_config.pretrain_model = configs.ModelType.BERT
task_config.finetune_task_name = configs.GlueTaskName.STSB
task_config.lora_adapt_type = configs.LoraAdaptType.ATTENTION_MLP
task_config.lora_init_scale = 1e-3
task_config.num_train_steps = 2000
task_config.train_batch_size = 16
task_config.eval_batch_size = 32
task_config.max_seq_length = 128
task_config.log_eval_steps = 100
task_config.learning_rate = 1e-4
task_config.decay_ratio = 0.1
task_config.save_step_points = [
    0,
    1,
    10,
    20,
    30,
    100,
    500,
    1000,
    1500,
    task_config.num_train_steps,
]
task_config.identifier = "intro"

try:
    finetune(task_config, seeds=[0])
except FileExistsError:
    print("Experiment already ran. Loading the model.")

experiment_path = logging_utils.get_experiment_path(task_config, seed=0)

task_config = logging_utils.get_task_config_from_json(experiment_path=experiment_path)
model_params = models.create_pretrain_model_from_config(task_config).params  # type: ignore
lora_model = models.create_lora_model_from_config(task_config, model_params)
final_lora_params = logging_utils.Checkpointer(experiment_path).load(
    task_config.num_train_steps
)
final_e2e = lora_model.apply({"params": final_lora_params})
final_e2e = cast(dict, final_e2e)
sv_vals_dict = {}
for k, v in final_e2e.items():
    sv_vals_dict[k] = np.linalg.svd(v, compute_uv=False)

series = np.array(list(sv_vals_dict.values()))[:, :10]
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
plot_utils.plot_series(ax, series, color="plasma", zoom=0.8, alpha=1.0, linewidth=0)
ax.set_yticks([])
ax.set_xlabel("\nSV Index", fontsize=14)
ax.set_ylabel("Adapted Layer", fontsize=14)
ax.set_yticks([])
# ax.set_zticks([])  # type: ignore
fig.savefig(
    os.path.join("../figures", "final_spectra.png"), dpi=500, bbox_inches="tight"
)
