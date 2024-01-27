import json
import os
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.ticker import MaxNLocator

import configs
import models
import utils

metric_dict = {
    "cola": "eval_matthews_correlation",
    "sst2": "eval_accuracy",
    "stsb": "eval_pearson",
    "qqp": "eval_accuracy",
    "mrpc": "eval_accuracy",
    "mnli": "eval_accuracy",
    "qnli": "eval_accuracy",
    "rte": "eval_accuracy",
}


def plot_series(
    ax,
    series,
    color="viridis",
    crossing_lines=False,
    x_points=None,
    y_points=None,
    zoom=0.8,
    elev=30,
    azim=-50,
    roll=0,
    alpha=1.0,
    linewidth=1.0,
):
    n_x_indices, n_y_indices = series.shape
    x_indices = np.arange(n_x_indices)
    y_indices = np.arange(n_y_indices)

    if x_points is None:
        x_points = x_indices
    if y_points is None:
        y_points = y_indices

    spectrum_verts = []

    for idx in x_indices:
        spectrum_verts.append(
            [
                (0, np.min(series) - 0.05),
                *zip(y_points, series[idx, :]),
                (y_points[-1], np.min(series) - 0.05),
            ]
        )

    spectrum_poly = PolyCollection(spectrum_verts)
    spectrum_poly.set_linewidth(linewidth)
    spectrum_poly.set_alpha(alpha)
    spectrum_poly.set_facecolor(
        plt.colormaps[color](np.linspace(0, 0.7, len(spectrum_verts)))  # type: ignore
    )
    spectrum_poly.set_edgecolor("black")

    ax.set_box_aspect(aspect=None, zoom=zoom)

    ax.add_collection3d(spectrum_poly, zs=x_indices, zdir="y")

    if crossing_lines:
        path_verts = []

        for idx in y_indices:
            path_verts.append([*zip(x_points, series[:, idx])])
        path_line = LineCollection(path_verts)
        path_line.set_linewidth(linewidth)
        path_line.set_edgecolor("black")
        ax.add_collection3d(path_line, zs=y_indices, zdir="x")

    ax.set_xlim(y_points[0], y_points[-1])
    ax.set_ylim(x_points[0], x_points[-1])
    ax.set_zlim(np.min(series) - 0.1, np.max(series) + 0.1)

    ax.view_init(elev, azim, roll)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def get_final_spectra(experiment_path: str):
    task_config = utils.get_task_config_from_json(experiment_path=experiment_path)
    model_params = models.create_pretrain_model_from_config(
        task_config, num_labels=1
    ).params  # type: ignore
    lora_model = models.create_lora_model_from_config(task_config, model_params)
    final_lora_params = utils.load_lora_params(
        experiment_path=experiment_path, step=task_config.num_train_steps
    )
    final_e2e = lora_model.apply({"params": final_lora_params})
    final_e2e = cast(dict, final_e2e)
    sv_vals_dict = {}
    for k, v in final_e2e.items():
        sv_vals_dict[k] = np.linalg.svd(v, compute_uv=False)

    return sv_vals_dict


def plot_final_spectra(experiment_path: str):
    sv_vals_dict = get_final_spectra(experiment_path=experiment_path)
    series = np.array(list(sv_vals_dict.values()))[:, :5]  # type: ignore
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_series(ax, series, color="plasma", zoom=0.8, alpha=1.0)
    ax.set_yticks([])
    ax.set_xlabel("\nSV Index", fontsize=14)
    ax.set_ylabel("Adapted Layer", fontsize=14)
    ax.set_yticks([])
    ax.set_zticks([])  # type: ignore
    return fig


def get_subspace_traj(experiment_path: str, rank: int):
    task_config = utils.get_task_config_from_json(experiment_path=experiment_path)
    model_params = models.create_pretrain_model_from_config(
        task_config, num_labels=1
    ).params  # type: ignore
    lora_model = models.create_lora_model_from_config(task_config, model_params)
    step_vals = np.array(task_config.save_step_points)

    flat_param_paths = lora_model.flat_params_shape_dict.keys()
    subspace_vals_dict = {k: [] for k in flat_param_paths}
    for step in step_vals:
        print(f"Loading step {step}")
        lora_params = utils.load_lora_params(experiment_path=experiment_path, step=step)
        e2e = lora_model.apply({"params": lora_params})
        e2e = cast(dict, e2e)

        for k, v in e2e.items():
            U, _, V = utils.svd(v)
            Ur, Vr = U[:, :rank], V[:, :rank]
            subspace_vals_dict[k].append((Ur, Vr))

    return subspace_vals_dict, step_vals


def get_cosine_angle_traj(experiment_path: str, rank: int, side="left"):
    subspace_vals_dict, step_vals = get_subspace_traj(experiment_path, rank)
    cosine_angle_vals_dict = {k: [] for k in subspace_vals_dict.keys()}
    for k, v in subspace_vals_dict.items():
        Ur_final, Vr_final = v[-1]
        for Ur, Vr in v:
            if side == "left":
                cosine_angle_vals_dict[k].append(utils.cosine_angle(Ur, Ur_final))
            elif side == "right":
                cosine_angle_vals_dict[k].append(utils.cosine_angle(Vr, Vr_final))
            else:
                raise ValueError(f"Invalid side {side}")
    return cosine_angle_vals_dict, step_vals


def plot_cosine_angle_traj(
    experiment_path: str, rank: int, task_config: configs.TaskConfig
):
    cosine_angle_vals_dict, step_vals = get_cosine_angle_traj(experiment_path, rank)
    series = np.array(list(cosine_angle_vals_dict.values()))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_series(
        ax,
        series,
        color="plasma",
        y_points=step_vals,
        zoom=0.8,
        elev=30,
        azim=-130,
        roll=0,
    )
    ax.set_xticks(np.linspace(0, task_config.num_train_steps, 5, dtype=int))
    ax.set_xlabel("\nIterations", fontsize=14)
    ax.set_ylabel("Adapted Layer", fontsize=14)
    ax.set_yticks([])
    ax.set_zlabel("Cosine Angle", fontsize=14)  # type: ignore
    return fig


def smooth(scalars, weight):
    """Exponential moving average."""
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed


def read_results(experiment_path):
    with open(os.path.join(experiment_path, "results.json")) as f:
        results = json.load(f)
    tags = results.keys()
    result_dict = {}
    for tag in tags:
        step_vals, value_vals = list(
            zip(*[(pair["step"], pair["value"]) for pair in results[tag]])
        )
        result_dict[tag] = (np.array(step_vals), np.array(value_vals))
    return result_dict


def get_fewshot_1024_results():
    experiment_dir = "experiments/glue_fewshot_1024"
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
                for entry in sorted(results[metric_dict[task]], key=lambda x: x["step"])
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


def plot_fewshot_1024_results():
    series, labels = get_fewshot_1024_results()

    def set_axis_style(ax, labels):
        ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels, fontsize=12)
        ax.set_xlim(0.25, len(labels) + 0.75)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.violinplot(series, showmeans=True, showextrema=False, widths=0.1)
    ax.yaxis.grid(True)
    ax.set_ylabel("$\Delta$", fontsize=12)
    ax.hlines(0, 0, 10)

    set_axis_style(ax, labels)
    return fig


def get_narrow_vs_wide_results():
    experiment_dir = "experiments/stsb_fewshot_1024_narrow_vs_wide"
    runs = os.listdir(experiment_dir)
    narrow_run = [x for x in runs if "rank=8" in x][0]
    wide_run = [x for x in runs if "rank=8" not in x][0]
    narrow_results = read_results(os.path.join(experiment_dir, narrow_run))
    wide_results = read_results(os.path.join(experiment_dir, wide_run))
    train_step_vals, narrow_train_loss_vals = narrow_results["train_loss"]
    _, wide_train_loss_vals = wide_results["train_loss"]
    eval_step_vals, narrow_eval_vals = narrow_results[metric_dict["stsb"]]
    _, wide_eval_vals = wide_results[metric_dict["stsb"]]
    return (
        train_step_vals,
        narrow_train_loss_vals,
        wide_train_loss_vals,
        eval_step_vals,
        narrow_eval_vals,
        wide_eval_vals,
    )


def plot_narrow_vs_wide_results():
    (
        train_step_vals,
        narrow_train_loss_vals,
        wide_train_loss_vals,
        eval_step_vals,
        narrow_eval_vals,
        wide_eval_vals,
    ) = get_narrow_vs_wide_results()
    fig, ax = plt.subplots(ncols=2, figsize=(9, 3))
    smooth_fn = lambda x: smooth(x, 0.95)
    linewidth = 4
    ax[0].plot(
        train_step_vals,
        smooth_fn(narrow_train_loss_vals),
        label="Narrow",
        linewidth=linewidth,
    )
    ax[0].plot(
        train_step_vals,
        smooth_fn(wide_train_loss_vals),
        label="Compressed",
        linewidth=linewidth,
    )
    ax[0].set_xlabel("Iteration", fontsize=14)
    ax[0].set_ylabel("Train Loss", fontsize=14)
    ax[0].legend(fontsize=14)
    ax[1].plot(
        eval_step_vals, smooth_fn(narrow_eval_vals), label="Narrow", linewidth=linewidth
    )
    ax[1].plot(
        eval_step_vals,
        smooth_fn(wide_eval_vals),
        label="Compressed",
        linewidth=linewidth,
    )
    ax[1].set_xlabel("Iteration", fontsize=14)
    ax[1].set_ylabel("Pearson Corr.", fontsize=14)
    ax[1].legend(fontsize=14)
    return fig


def get_fewshot_stsb_results():
    experiment_dir = "experiments/stsb_fewshot"
    sample_sizes = [int(x) for x in os.listdir(experiment_dir)]

    diff_dict = {}
    take_last = True

    def get_seed(run):
        return int(run.split("_")[-1].split("=")[1])

    for sample_size in sorted(sample_sizes):
        runs = [
            run for run in os.listdir(os.path.join(experiment_dir, str(sample_size)))
        ]
        runs_2 = [run for run in runs if "depth=2" in run]
        runs_3 = [run for run in runs if "depth=3" in run]
        runs_2_results = np.zeros(len(runs_2))
        runs_3_results = np.zeros(len(runs_3))
        for run in runs:
            seed = get_seed(run)
            with open(
                os.path.join(experiment_dir, str(sample_size), run, "results.json")
            ) as f:
                results = json.load(f)
            values = [
                entry["value"]
                for entry in sorted(results["eval_pearson"], key=lambda x: x["step"])
            ]
            if "depth=2" in run:
                runs_2_results[seed] = values[-1] if take_last else max(values)
            elif "depth=3" in run:
                runs_3_results[seed] = values[-1] if take_last else max(values)
        diff_dict[sample_size] = (
            runs_3_results,
            runs_2_results,
        )

    labels, series = list(zip(*diff_dict.items()))
    labels = list(labels)
    series = list(series)
    return series, labels


def plot_fewshot_stsb_results():
    eval_values, sample_sizes = get_fewshot_stsb_results()
    depth_3_values = [x[0] for x in eval_values]
    depth_2_values = [x[1] for x in eval_values]
    depth_3_values = np.array(depth_3_values)
    depth_2_values = np.array(depth_2_values)
    fig, ax = plt.subplots()
    ax.errorbar(
        sample_sizes,
        depth_3_values.mean(axis=1),
        fmt="-o",
        yerr=depth_3_values.std(axis=1) / 2,
        linewidth=4,
        capsize=6,
        markersize=7,
        label="Deep Compressed LoRA",
    )
    ax.errorbar(
        sample_sizes,
        depth_2_values.mean(axis=1),
        fmt="-o",
        yerr=depth_2_values.std(axis=1) / 2,
        linewidth=2,
        capsize=6,
        markersize=7,
        label="Vanilla LoRA",
    )
    ax.set_xscale("log")
    ax.set_xticks([16, 64, 256], labels=["16", "64", "256"], fontsize=14)
    ax.minorticks_off()
    ax.set_xlabel("# Training Examples", fontsize=14)
    ax.set_ylabel("Pearson Correlation", fontsize=14)
    ax.legend(fontsize=14)
    return fig
