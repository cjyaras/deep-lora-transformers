from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.ticker import MaxNLocator
from tqdm.auto import tqdm

import configs
import models
import utils


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
    for k, v in tqdm(final_e2e.items()):
        sv_vals_dict[k] = np.linalg.svd(v, compute_uv=False)

    return sv_vals_dict


def plot_final_spectra(experiment_path: str):
    sv_vals_dict = get_final_spectra(experiment_path=experiment_path)
    series = np.array(list(sv_vals_dict.values()))[:, :5]
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

        for k, v in tqdm(e2e.items()):
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
