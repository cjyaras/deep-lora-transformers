import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.ticker import MaxNLocator


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


def smooth(scalars, weight):
    "Exponential moving average."
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed
