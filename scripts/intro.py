import json
import os
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from dlt import configs, logging_utils, misc_utils, models, plot_utils
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


finetune(task_config, seeds=[0])
experiment_path = logging_utils.get_experiment_path(task_config, seed=0)

## Left figure

model_params = models.create_pretrain_model_from_config(task_config).params  # type: ignore
lora_model = models.create_lora_model_from_config(task_config, model_params)
final_lora_params = logging_utils.load_lora_params(
    experiment_path, task_config.num_train_steps
)
final_e2e = lora_model.apply({"params": final_lora_params})  # type: ignore
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
fig.savefig(
    os.path.join("../figures", "final_spectra.png"), dpi=500, bbox_inches="tight"
)

## Middle figure

rank = 8
step_vals = np.array(task_config.save_step_points)
flat_param_paths = lora_model.flat_params_shape_dict.keys()
subspace_vals_dict = {k: [] for k in flat_param_paths}

for step in step_vals:
    print(f"Loading step {step}")
    lora_params = logging_utils.load_lora_params(experiment_path, step)
    e2e = lora_model.apply({"params": lora_params})  # type: ignore
    e2e = cast(dict, e2e)

    for k, v in e2e.items():
        U, _, VT = np.linalg.svd(v)
        V = VT.T
        Ur, Vr = U[:, :rank], V[:, :rank]
        subspace_vals_dict[k].append((Ur, Vr))

side = "left"
cosine_angle_vals_dict = {k: [] for k in subspace_vals_dict.keys()}
for k, v in subspace_vals_dict.items():
    Ur_final, Vr_final = v[-1]
    for Ur, Vr in v:
        if side == "left":
            cosine_angle_vals_dict[k].append(misc_utils.cosine_angle(Ur, Ur_final))
        elif side == "right":
            cosine_angle_vals_dict[k].append(misc_utils.cosine_angle(Vr, Vr_final))
        else:
            raise ValueError(f"Invalid side {side}")

series = np.array(list(cosine_angle_vals_dict.values()))
filtered_series = series[series[:, 4] > 0.5]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
plot_utils.plot_series(
    ax,
    filtered_series,
    color="plasma",
    y_points=step_vals,
    zoom=0.8,
    elev=30,
    azim=-130,
    roll=0,
    linewidth=2.0,
    alpha=1.0,
    line_plot=True,
)
ax.set_xticks(np.linspace(0, task_config.num_train_steps, 5, dtype=int))
ax.set_xlabel("\nIteration", fontsize=14)
ax.set_ylabel("Adapted Layer", fontsize=14)
ax.set_yticks([])
ax.set_zlabel("Cosine Angle", fontsize=14)  # type: ignore
fig.savefig(
    os.path.join("../figures", "cosine_angle_traj.png"), dpi=500, bbox_inches="tight"
)

# ## Right figure

with open(os.path.join(experiment_path, "results.json")) as f:
    results = json.load(f)
step_vals, loss_vals = list(
    zip(*[(pair["step"], pair["value"]) for pair in results["train_loss"]])
)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(step_vals, plot_utils.smooth(loss_vals, 0.9))
ax.set_xlabel("Iteration", fontsize=20)
ax.set_ylabel("Train Loss", fontsize=20)
ax.set_ylim(0, 2)
fig.savefig(os.path.join("../figures", "train_loss.png"), dpi=500, bbox_inches="tight")
