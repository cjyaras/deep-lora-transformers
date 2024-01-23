from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.ticker import MaxNLocator
from tqdm.auto import tqdm

import models
import utils


def plot_sv_series(ax, series, color="viridis", spec_step=2):
    n_time_indices, n_sval_indices = series.shape
    time_indices = np.arange(n_time_indices)
    sval_indices = np.arange(n_sval_indices)

    spectrum_verts = []

    for idx in time_indices[::spec_step]:
        spectrum_verts.append(
            [
                (0, np.min(series) - 0.05),
                *zip(sval_indices, series[idx, :]),
                (n_sval_indices, np.min(series) - 0.05),
            ]
        )

    path_verts = []

    for idx in sval_indices:
        path_verts.append([*zip(time_indices, series[:, idx])])

    spectrum_poly = PolyCollection(spectrum_verts)
    spectrum_poly.set_alpha(0.8)
    spectrum_poly.set_facecolor(
        plt.colormaps[color](np.linspace(0, 0.7, len(spectrum_verts)))  # type: ignore
    )
    spectrum_poly.set_edgecolor("black")

    path_line = LineCollection(path_verts)
    path_line.set_linewidth(1)
    path_line.set_edgecolor("black")

    ax.set_box_aspect(aspect=None, zoom=0.8)

    ax.add_collection3d(spectrum_poly, zs=time_indices[::spec_step], zdir="y")
    ax.add_collection3d(path_line, zs=sval_indices, zdir="x")

    ax.set_xlim(0, n_sval_indices)
    ax.set_ylim(0, n_time_indices)
    ax.set_zlim(np.min(series) - 0.1, np.max(series) + 0.1)

    elev = 30
    azim = -50
    roll = 0
    ax.view_init(elev, azim, roll)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def get_final_spectra(experiment_path: str):
    task_config = utils.get_task_config_from_json(experiment_path=experiment_path)
    model_params = models.create_pretrain_model_from_config(
        task_config, num_labels=1
    ).params  # type: ignore
    lora_model = models.create_lora_model_from_config(task_config, model_params)
    # flat_param_paths = lora_model.flat_params_shape_dict.keys()
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
    plot_sv_series(ax, series, color="plasma", spec_step=1)
    ax.set_yticks([])
    ax.set_xlabel("\nSV Index", fontsize=14)
    ax.set_ylabel("Adapted Layer", fontsize=14)
    ax.set_yticks([])
    ax.set_zticks([])  # type: ignore
    plt.show()


def get_norm_trajectories(experiment_path: str):
    task_config = utils.get_task_config_from_json(experiment_path=experiment_path)
    model_params = models.create_pretrain_model_from_config(
        task_config, num_labels=1
    ).params  # type: ignore
    lora_model = models.create_lora_model_from_config(task_config, model_params)
    step_vals = np.array(task_config.save_step_points)

    flat_param_paths = lora_model.flat_params_shape_dict.keys()
    norm_vals_dict = {k: [] for k in flat_param_paths}
    for step in step_vals:
        print(f"Loading step {step}")
        lora_params = utils.load_lora_params(experiment_path=experiment_path, step=step)
        e2e = lora_model.apply({"params": lora_params})
        e2e = cast(dict, e2e)

        for k, v in tqdm(e2e.items()):
            norm_vals_dict[k].append(np.linalg.norm(v))

    for k, v in norm_vals_dict.items():
        norm_vals_dict[k] = np.array(v)  # type: ignore
    return norm_vals_dict, step_vals
