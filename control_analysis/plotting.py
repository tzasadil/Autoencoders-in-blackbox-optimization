from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from control_analysis.constants import BASELINE_COLOR, DEFAULT_COLOR
from control_analysis.transforms import default_groupby


_DEFAULT_BASELINES: pd.DataFrame | None = None


def set_default_baselines(baselines: pd.DataFrame | None) -> None:
    global _DEFAULT_BASELINES
    _DEFAULT_BASELINES = baselines


def _resolve_baselines(baselines: pd.DataFrame | None = None) -> pd.DataFrame:
    resolved = baselines if baselines is not None else _DEFAULT_BASELINES
    if resolved is None:
        raise RuntimeError("Baselines are not configured. Call set_default_baselines first.")
    return resolved


def bar(
    df: pd.DataFrame,
    x_name: str | Sequence[str] | None = None,
    y_name: str = "avg_rank",
    index_mapper: Callable[[object], object] | None = None,
    y_mapper: Callable[[object], object] | None = None,
    regr: bool = False,
    baseline_i: int | str = -1,
    x_ticklabel_mapper: Callable[[list[str]], list[str]] | None = None,
    print_table: str = "",
    title: str = "",
    table_path: str | os.PathLike[str] | None = None,
    baselines: pd.DataFrame | None = None,
) -> plt.Axes:
    from control_analysis.formatting import print_latex

    plotting_df = df.copy()
    if x_name is not None:
        plotting_df = default_groupby(plotting_df, x_name)
    plotting_df = plotting_df.sort_index()
    if index_mapper is not None:
        plotting_df.index = plotting_df.index.map(index_mapper)
        plotting_df = plotting_df.sort_index()
    if y_mapper is not None:
        plotting_df[y_name] = plotting_df[y_name].map(y_mapper)
    if print_table:
        if table_path is None:
            print(print_table)
        print_latex(plotting_df[y_name], output_path=table_path, heading=print_table)

    colors = [DEFAULT_COLOR for _ in range(len(plotting_df))]
    if baseline_i != -1:
        baseline_row = _resolve_baselines(baselines).loc[baseline_i]
        plotting_df.loc[f"baseline_{baseline_i}"] = baseline_row
        colors = colors + [BASELINE_COLOR]

    x_values = plotting_df.index.to_numpy()
    y_values = plotting_df[y_name]
    _, ax = plt.subplots()
    positions = np.arange(len(y_values))
    ax.bar(positions, y_values.to_numpy(), color=colors)
    ax.set_ylabel("Rank Percentile")
    x_label = plotting_df.index.name if x_name is None else x_name
    if x_label is not None:
        if isinstance(x_label, (list, tuple)):
            x_label = ", ".join(str(value) for value in x_label)
        ax.set_xlabel(str(x_label).replace("_", " ").title())

    tick_labels = [str(value) for value in x_values]
    if x_ticklabel_mapper is not None:
        tick_labels = x_ticklabel_mapper(tick_labels)
    if baseline_i != -1:
        tick_labels[-1] = "baseline"
        ax.axhline(y=y_values.iloc[-1], color=BASELINE_COLOR, linestyle="dotted")
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, size="small")
    if regr:
        xx = np.arange(len(y_values) - (1 if baseline_i != -1 else 0))
        regression_values = y_values[:-1] if baseline_i != -1 else y_values
        slope, intercept = np.polyfit(xx, regression_values, 1)
        ax.plot(xx, slope * xx + intercept, color="red", alpha=0.5)
    if title:
        ax.set_title(title)
    return ax


def two_layer_tics(ax: plt.Axes) -> None:
    plt.xticks(rotation=0, size="xx-small")
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(15)


def save_and_show(name: str, show: bool = True, output_dir: str | os.PathLike[str] = "graphs") -> Path:
    fig = plt.gcf()
    output_path = Path(output_dir) / f"{name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def xy_scatter(xs, ys, len_mod_s, len_mod_e, xdesc, ydesc):
    fig, ax = plt.subplots()
    if isinstance(xs, str) and xs == "index":
        xs, ys = list(
            zip(
                *[
                    (index, item)
                    for arr in ys.to_list()
                    for (index, item) in enumerate(arr[int(len(arr) * len_mod_s) : int(len(arr) * len_mod_e)])
                ]
            )
        )
    else:
        xs = np.array([item for arr in xs.to_list() for item in arr[int(len(arr) * len_mod_s) : int(len(arr) * len_mod_e)]])
        ys = np.array([item for arr in ys.to_list() for item in arr[int(len(arr) * len_mod_s) : int(len(arr) * len_mod_e)]])

    not_nan = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys = np.array(xs)[not_nan], np.array(ys)[not_nan]
    ax.scatter(xs, ys, marker=".")

    lr = scipy.stats.linregress(xs, ys)
    xx = np.linspace(np.min(xs), np.max(xs), num=100)
    ax.plot(xx, lr.slope * xx + lr.intercept, color="red")
    txt = [
        "r^2 = {:.3f}".format(lr.rvalue**2),
        "linreg start: {:.3f}".format(lr.intercept),
        "linreg end: {:.3f}".format(lr.slope * np.max(xs) + lr.intercept),
    ]
    ax.annotate("\n".join(txt), (0.8 * np.max(xs), 0.8 * np.max(ys)))
    ax.set_xlabel(xdesc.title())
    ax.set_ylabel(ydesc.title())
    fig.show()
    return ax