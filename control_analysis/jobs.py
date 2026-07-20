from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from itertools import repeat
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from control_analysis.constants import EVAL_WINDOW_FUNC_GROUPS, FUNC_GROUP_LABELS, PLAIN_DOE_MODEL, PRIMARY_DOE_MODEL, SELECTOR_BASELINE_MODELS, display_model_label
try:
    from control_analysis.oracle_table import load_selector_baseline_ranks
except ModuleNotFoundError:
    def load_selector_baseline_ranks() -> pd.DataFrame | None:
        return None
from control_analysis.formatting import write_dataframe_tabular
from control_analysis.models import ControlDataBundle, EvalWindowGraphSpec
from control_analysis.plotting import bar, save_and_show, two_layer_tics
from control_analysis.stats import write_stats_report
from control_analysis.transforms import add_func_group, default_groupby, improvement_percent


_WORKER_BUNDLE: ControlDataBundle | None = None
COMPARISON_MODELS = [
    "none",
    "gp",
    "nn3",
    "elm100",
    PLAIN_DOE_MODEL,
    PRIMARY_DOE_MODEL,
    "fitloss",
]
_MODEL_DISPLAY_ORDER = COMPARISON_MODELS
RANK_METRIC = "final_rank"
RANK_SOURCE = "last_rank"
RANK_LABEL = "Average rank percentile"
RANK_TITLE = "Average rank"
DISTANCE_REFERENCE_DIM = 2


def _nanmean_array(values: np.ndarray | list[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or np.all(np.isnan(array)):
        return np.nan
    return float(np.nanmean(array))


def _ordered_labels(values: pd.Index | list[object], preferred: list[object]) -> list[object]:
    present = list(values)
    ordered = [value for value in preferred if value in present]
    remainder = sorted((value for value in present if value not in ordered), key=str)
    return ordered + remainder


def _table_column_format(column_count: int, first_column_alignment: str = "l") -> str:
    return first_column_alignment + ("r" * max(0, column_count - 1))


def _write_plot_table(
    df: pd.DataFrame,
    output_name: str,
    output_dir: str | os.PathLike[str],
    first_column_alignment: str = "l",
) -> Path:
    output_path = Path(output_dir) / f"{output_name}.tex"
    return write_dataframe_tabular(
        df,
        output_path=output_path,
        column_format=_table_column_format(len(df.columns), first_column_alignment),
    )


def _write_series_plot_table(
    series: pd.Series,
    output_name: str,
    output_dir: str | os.PathLike[str],
    index_label: str,
    value_label: str,
    first_column_alignment: str = "l",
) -> Path:
    export_df = series.reset_index()
    export_df = export_df.rename(columns={export_df.columns[0]: index_label, export_df.columns[1]: value_label})
    return _write_plot_table(
        export_df,
        output_name=output_name,
        output_dir=output_dir,
        first_column_alignment=first_column_alignment,
    )


def _plot_metric_bar(
    summary: pd.DataFrame,
    value_column: str,
    title: str,
    ylabel: str,
    output_name: str,
    output_dir: str | os.PathLike[str],
    index_mapper=None,
    table_index_label: str | None = None,
    table_value_label: str | None = None,
) -> Path | None:
    if summary.empty:
        return None
    plotting_df = summary[[value_column]].copy()
    plotting_df = plotting_df.dropna(subset=[value_column])
    if plotting_df.empty:
        return None
    if index_mapper is not None:
        plotting_df.index = plotting_df.index.map(index_mapper)
        plotting_df = plotting_df.sort_index()
    index_column_name = table_index_label or (
        str(plotting_df.index.name).replace("_", " ").title()
        if plotting_df.index.name is not None
        else "Label"
    )
    value_column_name = table_value_label or ylabel
    export_df = plotting_df.reset_index()
    export_df = export_df.rename(columns={export_df.columns[0]: index_column_name, value_column: value_column_name})
    _write_plot_table(export_df, output_name=output_name, output_dir=output_dir)
    ax = bar(plotting_df, y_name=value_column)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if "func_group" in output_name or "selector_baseline" in output_name:
        two_layer_tics(ax)
    return save_and_show(output_name, show=False, output_dir=output_dir)


def _plot_generation_progress(
    summary: pd.DataFrame,
    value_column: str,
    title: str,
    ylabel: str,
    output_name: str,
    output_dir: str | os.PathLike[str],
    table_value_label: str | None = None,
) -> Path | None:
    if summary.empty:
        return None
    export_df = summary[["generation_fraction", value_column]].rename(
        columns={
            "generation_fraction": "Generation fraction",
            value_column: table_value_label or ylabel,
        }
    )
    _write_plot_table(export_df, output_name=output_name, output_dir=output_dir, first_column_alignment="r")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(summary["generation_fraction"], summary[value_column], color="forestgreen", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Fraction of generations")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    return save_and_show(output_name, show=False, output_dir=output_dir)


def _flatten_metric_pairs(df: pd.DataFrame, x_column: str, y_column: str) -> tuple[np.ndarray, np.ndarray]:
    x_arrays: list[np.ndarray] = []
    y_arrays: list[np.ndarray] = []
    for x_values, y_values in zip(df[x_column].tolist(), df[y_column].tolist()):
        x_array = np.asarray(x_values, dtype=float)
        y_array = np.asarray(y_values, dtype=float)
        if x_array.ndim != 1 or y_array.ndim != 1 or x_array.size == 0 or y_array.size == 0:
            continue
        pair_count = min(x_array.size, y_array.size)
        if pair_count == 0:
            continue
        x_arrays.append(x_array[:pair_count])
        y_arrays.append(y_array[:pair_count])
    if not x_arrays:
        return np.array([], dtype=float), np.array([], dtype=float)
    xs = np.concatenate(x_arrays)
    ys = np.concatenate(y_arrays)
    finite_mask = np.isfinite(xs) & np.isfinite(ys)
    return xs[finite_mask], ys[finite_mask]


def _flatten_distance_correlation_pairs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_arrays: list[np.ndarray] = []
    y_arrays: list[np.ndarray] = []
    dim_arrays: list[np.ndarray] = []
    for dim, x_values, y_values in zip(df["dim"].tolist(), df["dists"].tolist(), df["spearman_corr"].tolist()):
        x_array = np.asarray(x_values, dtype=float)
        y_array = np.asarray(y_values, dtype=float)
        if x_array.ndim != 1 or y_array.ndim != 1 or x_array.size == 0 or y_array.size == 0:
            continue
        pair_count = min(x_array.size, y_array.size)
        if pair_count == 0:
            continue
        x_arrays.append(x_array[:pair_count])
        y_arrays.append(y_array[:pair_count])
        dim_arrays.append(np.full(pair_count, int(dim), dtype=int))
    if not x_arrays:
        empty = np.array([], dtype=float)
        return empty, empty, np.array([], dtype=int)
    xs = np.concatenate(x_arrays)
    ys = np.concatenate(y_arrays)
    dims = np.concatenate(dim_arrays)
    finite_mask = np.isfinite(xs) & np.isfinite(ys)
    return xs[finite_mask], ys[finite_mask], dims[finite_mask]


def _normalize_distances_to_reference_dim(xs: np.ndarray, dim: int) -> np.ndarray:
    return xs * np.sqrt(DISTANCE_REFERENCE_DIM / dim)


def _build_distance_correlation_summary(xs: np.ndarray, ys: np.ndarray) -> pd.DataFrame:
    regression = stats.linregress(xs, ys)
    spearman = stats.spearmanr(xs, ys)
    thresholds = [0.05, 0.10, 0.20]
    rows = [
        ("Pair count", float(xs.size)),
        (f"Distance min (normalized to {DISTANCE_REFERENCE_DIM}D)", float(np.min(xs))),
        (f"Distance mean (normalized to {DISTANCE_REFERENCE_DIM}D)", float(np.mean(xs))),
        (f"Distance median (normalized to {DISTANCE_REFERENCE_DIM}D)", float(np.median(xs))),
        (f"Distance p90 (normalized to {DISTANCE_REFERENCE_DIM}D)", float(np.quantile(xs, 0.90))),
        (f"Distance max (normalized to {DISTANCE_REFERENCE_DIM}D)", float(np.max(xs))),
        ("Correlation min", float(np.min(ys))),
        ("Correlation mean", float(np.mean(ys))),
        ("Correlation median", float(np.median(ys))),
        ("Correlation p90", float(np.quantile(ys, 0.90))),
        ("Correlation max", float(np.max(ys))),
        ("Linear slope", float(regression.slope)),
        ("Linear intercept", float(regression.intercept)),
        ("Pearson r", float(regression.rvalue)),
        ("R^2", float(regression.rvalue**2)),
        ("Linear p-value", float(regression.pvalue)),
        ("Spearman rho", float(spearman.statistic)),
        ("Spearman p-value", float(spearman.pvalue)),
    ]
    rows.extend(
        (
            f"Share normalized distance <= {threshold:0.2f}",
            float(np.mean(xs <= threshold)),
        )
        for threshold in thresholds
    )
    return pd.DataFrame(rows, columns=["Statistic", "Value"])


def _render_distance_correlation_graphs(
    df_og: pd.DataFrame,
    output_dir: str | os.PathLike[str],
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    doe_rows = df_og[df_og["model"] == PRIMARY_DOE_MODEL].copy()
    if doe_rows.empty or "dists" not in doe_rows.columns or "spearman_corr" not in doe_rows.columns:
        return {}

    results: dict[str, Path] = {}
    dim_specs: list[tuple[int | None, str, str]] = [
        (2, "dim_2", "Dimension 2"),
        (5, "dim_5", "Dimension 5"),
        (10, "dim_10", "Dimension 10"),
        (None, "All", "All dimensions"),
    ]
    for dim, output_name, title in dim_specs:
        filtered = doe_rows if dim is None else doe_rows[doe_rows["dim"] == dim]
        xs, ys, dims = _flatten_distance_correlation_pairs(filtered)
        if xs.size < 2:
            continue
        if dim is None:
            xs = xs * np.sqrt(DISTANCE_REFERENCE_DIM / dims)
        else:
            xs = _normalize_distances_to_reference_dim(xs, dim)
        regression = stats.linregress(xs, ys)
        figure, ax = plt.subplots()
        ax.scatter(xs, ys, marker=".", alpha=0.35, s=9)
        line_x = np.linspace(float(np.min(xs)), float(np.max(xs)), num=100)
        ax.plot(line_x, regression.slope * line_x + regression.intercept, color="red")
        ax.set_title(title)
        ax.set_xlabel(f"Latent space Euclidean distance normalized to {DISTANCE_REFERENCE_DIM}D")
        ax.set_ylabel("Spearman rank correlation")
        summary_text = "\n".join(
            [
                f"R^2 = {regression.rvalue**2:0.3f}",
                f"slope = {regression.slope:0.3f}",
                f"pairs = {xs.size}",
            ]
        )
        ax.annotate(summary_text, (0.65, 0.05), xycoords="axes fraction")
        figure_path = save_and_show(output_name, show=False, output_dir=output_path)
        summary_df = _build_distance_correlation_summary(xs, ys)
        _write_plot_table(summary_df, output_name=output_name, output_dir=output_path)
        results[output_name] = figure_path
    return results


def _build_problem_metric_summary(df_og: pd.DataFrame) -> pd.DataFrame:
    analysis_df = add_func_group(df_og)
    analysis_df[RANK_METRIC] = analysis_df["ranks"].apply(lambda values: values[-1])
    analysis_df["avg_spearman_corr"] = analysis_df["spearman_corr"].apply(_nanmean_array)
    analysis_df["avg_spearman_pval"] = analysis_df["spearman_pval"].apply(_nanmean_array)
    if "selected_spread_ratio" in analysis_df.columns:
        analysis_df["avg_selected_spread_ratio"] = analysis_df["selected_spread_ratio"].apply(_nanmean_array)
    else:
        analysis_df["avg_selected_spread_ratio"] = np.nan
    if "selected_radius_ratio" in analysis_df.columns:
        analysis_df["avg_selected_radius_ratio"] = analysis_df["selected_radius_ratio"].apply(_nanmean_array)
    else:
        analysis_df["avg_selected_radius_ratio"] = np.nan
    if "selection_quality_gap" in analysis_df.columns:
        analysis_df["avg_selection_quality_gap"] = analysis_df["selection_quality_gap"].apply(_nanmean_array)
    else:
        analysis_df["avg_selection_quality_gap"] = np.nan
    if "oracle_regret" in analysis_df.columns:
        analysis_df["avg_oracle_regret"] = analysis_df["oracle_regret"].apply(_nanmean_array)
    else:
        analysis_df["avg_oracle_regret"] = np.nan

    return analysis_df.groupby(
        ["func_group", "func_group_key", "function", "instance", "dim", "model", "true_ratio"],
        as_index=False,
    ).agg(
        final_rank=(RANK_METRIC, "mean"),
        avg_spearman_corr=("avg_spearman_corr", "mean"),
        avg_spearman_pval=("avg_spearman_pval", "mean"),
        avg_selected_spread_ratio=("avg_selected_spread_ratio", "mean"),
        avg_selected_radius_ratio=("avg_selected_radius_ratio", "mean"),
        avg_selection_quality_gap=("avg_selection_quality_gap", "mean"),
        avg_oracle_regret=("avg_oracle_regret", "mean"),
        elapsed_time=("elapsed_time", "sum"),
    )


def _build_generation_fraction_summary(df_og: pd.DataFrame, array_column: str, value_column: str) -> pd.DataFrame:
    doe_rows = df_og[df_og["model"] == PRIMARY_DOE_MODEL]
    if doe_rows.empty or array_column not in doe_rows.columns:
        return pd.DataFrame(columns=["generation_fraction", value_column])

    arrays = []
    for values in doe_rows[array_column].tolist():
        array = np.asarray(values, dtype=float)
        if array.ndim == 1 and array.size > 0:
            arrays.append(array)
    if not arrays:
        return pd.DataFrame(columns=["generation_fraction", value_column])

    stacked = np.vstack(arrays)
    num_generations = stacked.shape[1]
    return pd.DataFrame(
        {
            "generation_fraction": np.arange(1, num_generations + 1, dtype=float) / num_generations,
            value_column: np.nanmean(stacked, axis=0),
        }
    )


def _render_model_breakdown_graphs(per_problem: pd.DataFrame, output_dir: str | os.PathLike[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    metric_specs = [
        ("avg_spearman_corr", "surr_correlation", "Average Spearman correlation", "Surrogate correlation", True),
        ("avg_spearman_pval", "surr_pval", "Average Spearman p-value", "Surrogate p-value", True),
        (RANK_METRIC, "avg_rank", RANK_LABEL, RANK_TITLE, False),
    ]

    for dim in [2, 5, 10, None]:
        dim_label = "all_dims" if dim is None else f"dim_{dim}"
        title_suffix = "all dims" if dim is None else f"dim={dim}"
        filtered = per_problem if dim is None else per_problem[per_problem["dim"] == dim]
        if filtered.empty:
            continue
        model_summary = filtered.groupby("model", as_index=False).agg(
            avg_spearman_corr=("avg_spearman_corr", "mean"),
            avg_spearman_pval=("avg_spearman_pval", "mean"),
            final_rank=(RANK_METRIC, "mean"),
        )
        order = _ordered_labels(model_summary["model"].tolist(), _MODEL_DISPLAY_ORDER)
        model_summary = model_summary.set_index("model").loc[order]

        for value_column, slug, ylabel, title_prefix, surrogate_only in metric_specs:
            plotting_df = model_summary
            if surrogate_only:
                plotting_df = plotting_df.loc[plotting_df.index != "none"]
            path = _plot_metric_bar(
                plotting_df,
                value_column=value_column,
                title=f"{title_prefix}, {title_suffix}",
                ylabel=ylabel,
                output_name=f"{slug}_{dim_label}",
                output_dir=output_dir,
                index_mapper=display_model_label,
                table_index_label="Model",
                table_value_label=ylabel,
            )
            if path is not None:
                paths[f"{slug}_{dim_label}"] = path

    return paths


def _render_doe_focus_graphs(per_problem: pd.DataFrame, df_og: pd.DataFrame, output_dir: str | os.PathLike[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    doe_summary = per_problem[per_problem["model"] == PRIMARY_DOE_MODEL].copy()
    if doe_summary.empty:
        return paths

    ordered_groups = [label for _, _, _, label in FUNC_GROUP_LABELS]
    doe_by_func_group = (
        doe_summary.groupby(["func_group", "func_group_key"], as_index=False)
        .agg(
            final_rank=(RANK_METRIC, "mean"),
            avg_spearman_corr=("avg_spearman_corr", "mean"),
            avg_selected_spread_ratio=("avg_selected_spread_ratio", "mean"),
            avg_selected_radius_ratio=("avg_selected_radius_ratio", "mean"),
            avg_selection_quality_gap=("avg_selection_quality_gap", "mean"),
            avg_oracle_regret=("avg_oracle_regret", "mean"),
        )
        .set_index("func_group")
        .reindex(ordered_groups)
        .dropna(how="all")
    )
    doe_by_dim = (
        doe_summary.groupby("dim", as_index=False)
        .agg(
            final_rank=(RANK_METRIC, "mean"),
            avg_spearman_corr=("avg_spearman_corr", "mean"),
            avg_selected_spread_ratio=("avg_selected_spread_ratio", "mean"),
            avg_selected_radius_ratio=("avg_selected_radius_ratio", "mean"),
            avg_selection_quality_gap=("avg_selection_quality_gap", "mean"),
            avg_oracle_regret=("avg_oracle_regret", "mean"),
        )
        .sort_values("dim")
        .set_index("dim")
    )
    doe_spearman_by_fraction = _build_generation_fraction_summary(df_og, "spearman_corr", "avg_spearman_corr")
    doe_spread_ratio_by_fraction = _build_generation_fraction_summary(df_og, "selected_spread_ratio", "avg_selected_spread_ratio")
    doe_radius_ratio_by_fraction = _build_generation_fraction_summary(df_og, "selected_radius_ratio", "avg_selected_radius_ratio")
    doe_quality_gap_by_fraction = _build_generation_fraction_summary(df_og, "selection_quality_gap", "avg_selection_quality_gap")
    doe_oracle_regret_by_fraction = _build_generation_fraction_summary(df_og, "oracle_regret", "avg_oracle_regret")

    bar_specs = [
        (doe_by_func_group, RANK_METRIC, "DOE average rank by function group", RANK_LABEL, "doe_avg_rank_by_func_group"),
        (doe_by_dim, RANK_METRIC, "DOE average rank by dimension", RANK_LABEL, "doe_avg_rank_by_dim"),
        (doe_by_func_group, "avg_spearman_corr", "DOE surrogate correlation by function group", "Average Spearman correlation", "doe_spearman_by_func_group"),
        (doe_by_dim, "avg_spearman_corr", "DOE surrogate correlation by dimension", "Average Spearman correlation", "doe_spearman_by_dim"),
        (doe_by_func_group, "avg_selected_spread_ratio", "DOE selected-set spread ratio by function group", "Selected/all pairwise spread", "doe_selected_spread_ratio_by_func_group"),
        (doe_by_dim, "avg_selected_spread_ratio", "DOE selected-set spread ratio by dimension", "Selected/all pairwise spread", "doe_selected_spread_ratio_by_dim"),
        (doe_by_func_group, "avg_selected_radius_ratio", "DOE selected-set radius ratio by function group", "Selected/all mean radius", "doe_selected_radius_ratio_by_func_group"),
        (doe_by_dim, "avg_selected_radius_ratio", "DOE selected-set radius ratio by dimension", "Selected/all mean radius", "doe_selected_radius_ratio_by_dim"),
        (doe_by_func_group, "avg_oracle_regret", "DOE oracle regret by function group", "Mean regret vs oracle top-k", "doe_oracle_regret_by_func_group"),
        (doe_by_dim, "avg_oracle_regret", "DOE oracle regret by dimension", "Mean regret vs oracle top-k", "doe_oracle_regret_by_dim"),
    ]

    for summary, value_column, title, ylabel, output_name in bar_specs:
        table_index_label = "Function group" if "func_group" in output_name else "Dimension"
        path = _plot_metric_bar(
            summary,
            value_column=value_column,
            title=title,
            ylabel=ylabel,
            output_name=output_name,
            output_dir=output_dir,
            table_index_label=table_index_label,
            table_value_label=ylabel,
        )
        if path is not None:
            paths[output_name] = path

    generation_path = _plot_generation_progress(
        doe_spearman_by_fraction,
        value_column="avg_spearman_corr",
        title="DOE surrogate correlation by generation progress",
        ylabel="Average Spearman correlation",
        output_name="doe_spearman_by_generation_fraction",
        output_dir=output_dir,
        table_value_label="Average Spearman correlation",
    )
    if generation_path is not None:
        paths["doe_spearman_by_generation_fraction"] = generation_path

    generation_specs = [
        (doe_spread_ratio_by_fraction, "avg_selected_spread_ratio", "DOE selected-set spread ratio by generation progress", "Selected/all pairwise spread", "doe_selected_spread_ratio_by_generation_fraction"),
        (doe_radius_ratio_by_fraction, "avg_selected_radius_ratio", "DOE selected-set radius ratio by generation progress", "Selected/all mean radius", "doe_selected_radius_ratio_by_generation_fraction"),
        (doe_quality_gap_by_fraction, "avg_selection_quality_gap", "DOE selection quality gap by generation progress", "Rejected mean - selected mean", "doe_selection_quality_gap_by_generation_fraction"),
        (doe_oracle_regret_by_fraction, "avg_oracle_regret", "DOE oracle regret by generation progress", "Mean regret vs oracle top-k", "doe_oracle_regret_by_generation_fraction"),
    ]
    for summary, value_column, title, ylabel, output_name in generation_specs:
        path = _plot_generation_progress(
            summary,
            value_column=value_column,
            title=title,
            ylabel=ylabel,
            output_name=output_name,
            output_dir=output_dir,
            table_value_label=ylabel,
        )
        if path is not None:
            paths[output_name] = path

    return paths


def _render_selector_baseline_graph(df_og: pd.DataFrame, output_dir: str | os.PathLike[str]) -> Path | None:
    selector_models = [
        "oracle",
        "cluster_best_half_oracle",
        "none",
        PRIMARY_DOE_MODEL,
        "cluster_random_half",
        "negative_oracle",
    ]
    selector_df = df_og[df_og["model"].isin(selector_models)].copy()
    if selector_df.empty:
        summary = load_selector_baseline_ranks()
        if summary is None:
            return None
        summary = summary.rename(index=display_model_label)
    else:
        if "avg_rank" not in selector_df.columns:
            selector_df["avg_rank"] = selector_df["ranks"].apply(np.mean)
        summary = selector_df.groupby("model", as_index=False).agg(avg_rank=("avg_rank", "mean"))
        order = _ordered_labels(summary["model"].tolist(), selector_models)
        summary = summary.set_index("model").loc[order]
        summary = summary.rename(index=display_model_label)
    oracle_export = summary.reset_index().rename(
        columns={"index": "Model", "avg_rank": "Average rank percentile"}
    )
    write_dataframe_tabular(
        oracle_export,
        Path(output_dir).parent / "oracle_experiment.tex",
        "lr",
    )
    return _plot_metric_bar(
        summary,
        value_column="avg_rank",
        title="Selector diagnostic comparison, all dims",
        ylabel=RANK_LABEL,
        output_name="selector_baseline_avg_rank_all_dims",
        output_dir=output_dir,
        table_index_label="Model",
        table_value_label=RANK_LABEL,
    )


def _render_runtime_graph(df_og: pd.DataFrame, output_dir: str | os.PathLike[str]) -> Path | None:
    runtime_summary = df_og.groupby("model", as_index=False).agg(total_elapsed_time=("elapsed_time", "sum"))
    if runtime_summary.empty:
        return None
    order = _ordered_labels(runtime_summary["model"].tolist(), _MODEL_DISPLAY_ORDER)
    runtime_summary = runtime_summary.set_index("model").loc[order]
    return _plot_metric_bar(
        runtime_summary,
        value_column="total_elapsed_time",
        title="Total runtime by model",
        ylabel="Total runtime (s)",
        output_name="total_runtime_by_model",
        output_dir=output_dir,
        index_mapper=display_model_label,
        table_index_label="Model",
        table_value_label="Total runtime (s)",
    )


def run_doe_group_analysis(df_og: pd.DataFrame, output_dir: str | os.PathLike[str] = "graphs/avgs") -> dict[str, Path | pd.DataFrame]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    analysis_df = add_func_group(df_og)
    analysis_df = analysis_df[analysis_df["model"].isin(COMPARISON_MODELS)].copy()
    analysis_df[RANK_METRIC] = analysis_df["ranks"].apply(lambda values: values[-1])
    per_problem_metrics = _build_problem_metric_summary(analysis_df)

    per_problem = analysis_df.groupby(
        ["func_group", "func_group_key", "dim", "function", "instance", "model"],
        as_index=False,
    ).agg(final_rank=(RANK_METRIC, "mean"))

    group_summary = per_problem.groupby(["func_group", "func_group_key", "dim", "model"], as_index=False).agg(
        final_rank=("final_rank", "mean"),
        problems=("final_rank", "size"),
    )

    doe_summary = group_summary[group_summary["model"] == PRIMARY_DOE_MODEL].rename(columns={"final_rank": "doe_final_rank"})
    baseline_summary = group_summary[group_summary["model"] == "none"].rename(columns={"final_rank": "baseline_final_rank"})
    peer_summary = (
        group_summary[group_summary["model"] != PRIMARY_DOE_MODEL]
        .groupby(["func_group", "func_group_key", "dim"], as_index=False)
        .agg(peer_final_rank=("final_rank", "mean"), best_non_doe_final_rank=("final_rank", "min"))
    )

    doe_group_eval = doe_summary.merge(
        baseline_summary[["func_group", "dim", "baseline_final_rank"]],
        on=["func_group", "dim"],
        how="left",
    ).merge(peer_summary, on=["func_group", "func_group_key", "dim"], how="left")

    rank_within_group = group_summary.copy()
    rank_within_group["model_rank"] = rank_within_group.groupby(["func_group", "dim"])["final_rank"].rank(
        method="dense", ascending=False
    )
    doe_rank = rank_within_group[rank_within_group["model"] == PRIMARY_DOE_MODEL][["func_group", "dim", "model_rank"]]
    doe_group_eval = doe_group_eval.merge(doe_rank, on=["func_group", "dim"], how="left")

    doe_group_eval["vs_baseline"] = doe_group_eval["doe_final_rank"] - doe_group_eval["baseline_final_rank"]
    doe_group_eval["vs_peer_mean"] = doe_group_eval["doe_final_rank"] - doe_group_eval["peer_final_rank"]
    doe_group_eval["vs_best_non_doe"] = doe_group_eval["doe_final_rank"] - doe_group_eval["best_non_doe_final_rank"]
    doe_group_eval = doe_group_eval.sort_values(["dim", "func_group_key"])

    doe_by_dim = doe_group_eval.groupby("dim", as_index=False).agg(
        doe_final_rank=("doe_final_rank", "mean"),
        vs_baseline=("vs_baseline", "mean"),
        vs_peer_mean=("vs_peer_mean", "mean"),
        vs_best_non_doe=("vs_best_non_doe", "mean"),
        mean_rank=("model_rank", "mean"),
    ).sort_values("dim")

    doe_by_func_group = doe_group_eval.groupby(["func_group", "func_group_key"], as_index=False).agg(
        doe_final_rank=("doe_final_rank", "mean"),
        vs_baseline=("vs_baseline", "mean"),
        vs_peer_mean=("vs_peer_mean", "mean"),
        vs_best_non_doe=("vs_best_non_doe", "mean"),
        mean_rank=("model_rank", "mean"),
    ).sort_values("func_group_key")

    doe_by_dim_export = doe_by_dim.rename(
        columns={"dim": "Dimension", "doe_final_rank": "DOE avg. rank", "vs_baseline": "DOE - baseline", "mean_rank": "DOE rank"}
    )[["Dimension", "DOE avg. rank", "DOE - baseline", "DOE rank"]]
    doe_by_func_group_export = doe_by_func_group.rename(
        columns={"func_group": "Function group", "doe_final_rank": "DOE avg. rank", "vs_baseline": "DOE - baseline", "mean_rank": "DOE rank"}
    )[["Function group", "DOE avg. rank", "DOE - baseline", "DOE rank"]]

    dim_table_path = write_dataframe_tabular(doe_by_dim_export, output_path / "doe_by_dim_summary.tex", "rccc")
    group_table_path = write_dataframe_tabular(doe_by_func_group_export, output_path / "doe_by_func_group_summary.tex", "lccc")

    heatmap_path = output_path / "doe_group_heatmaps.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    preferred_labels = [label for _, _, _, label in FUNC_GROUP_LABELS]
    vs_baseline = doe_group_eval.pivot(index="func_group", columns="dim", values="vs_baseline")
    model_rank = doe_group_eval.pivot(index="func_group", columns="dim", values="model_rank")
    ordered_labels = _ordered_labels(vs_baseline.index, preferred_labels)
    sns.heatmap(
        vs_baseline.loc[ordered_labels],
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        ax=axes[0],
    )
    axes[0].set_title("DOE minus baseline\npositive = DOE better")
    axes[0].set_xlabel("Dimension")
    axes[0].set_ylabel("Function group")
    sns.heatmap(
        model_rank.loc[ordered_labels],
        annot=True,
        fmt=".1f",
        cmap="YlGn_r",
        vmin=1,
        vmax=4,
        ax=axes[1],
    )
    axes[1].set_title("DOE rank among model kinds\n1 = best")
    axes[1].set_xlabel("Dimension")
    axes[1].set_ylabel("Function group")
    fig.savefig(heatmap_path, bbox_inches="tight")
    plt.close(fig)
    heatmap_export = doe_group_eval.rename(
        columns={
            "func_group": "Function group",
            "dim": "Dimension",
            "vs_baseline": "DOE - baseline",
            "model_rank": "DOE rank",
        }
    )[["Function group", "Dimension", "DOE - baseline", "DOE rank"]]
    heatmap_table_path = _write_plot_table(
        heatmap_export,
        output_name="doe_group_heatmaps",
        output_dir=output_path,
    )

    model_breakdown_paths = _render_model_breakdown_graphs(per_problem_metrics, output_dir=output_path)
    doe_focus_paths = _render_doe_focus_graphs(per_problem_metrics, analysis_df, output_dir=output_path)
    selector_baseline_path = _render_selector_baseline_graph(df_og, output_dir=output_path)
    runtime_path = _render_runtime_graph(analysis_df, output_dir=output_path)
    dist_corr_paths = _render_distance_correlation_graphs(analysis_df, output_dir=output_path.parent / "dist_corr")

    result: dict[str, Path | pd.DataFrame] = {
        "dim_table_path": dim_table_path,
        "group_table_path": group_table_path,
        "heatmap_path": heatmap_path,
        "heatmap_table_path": heatmap_table_path,
        "doe_by_dim": doe_by_dim_export,
        "doe_by_func_group": doe_by_func_group_export,
    }
    result.update({key: value for key, value in model_breakdown_paths.items()})
    result.update({key: value for key, value in doe_focus_paths.items()})
    result.update({key: value for key, value in dist_corr_paths.items()})
    if selector_baseline_path is not None:
        result["selector_baseline_avg_rank_all_dims"] = selector_baseline_path
    if runtime_path is not None:
        result["total_runtime_by_model"] = runtime_path
    return result


def _build_eval_window_graph_specs() -> list[EvalWindowGraphSpec]:
    specs = []
    for frac_eval_limit in [5, 2, 1]:
        for dim in [2, 5, 10, None]:
            for func_start, func_end, description in EVAL_WINDOW_FUNC_GROUPS:
                specs.append(EvalWindowGraphSpec(frac_eval_limit=frac_eval_limit, dim=dim, func_start=func_start, func_end=func_end, description=description))
    return specs


def render_eval_window_graph(bundle: ControlDataBundle, spec: EvalWindowGraphSpec, output_dir: str | os.PathLike[str] = "graphs") -> Path | None:
    df = bundle.df_og.copy()
    df = df[~df["model"].isin(SELECTOR_BASELINE_MODELS)].copy()
    title_parts = []
    eval_limit_factor = 250.0
    if spec.description:
        title_parts.append(spec.description)
        df = df[(df["function"] >= spec.func_start) & (df["function"] <= spec.func_end)]
    if spec.dim is not None:
        title_parts.append(f"dim {spec.dim}")
        df = df[df["dim"] == spec.dim]
    else:
        title_parts.append("all dims")
    if spec.frac_eval_limit != 1:
        title_parts.append(f"first 1/{spec.frac_eval_limit} evaluations")
        eval_limit_factor = 250.0 / spec.frac_eval_limit
    if df.empty:
        return None

    title = ", ".join(title_parts)
    graph_name = title.replace("/", "")
    table_path = Path(output_dir) / f"{graph_name}.tex"

    df["improvement_percent"] = df["vals"].apply(improvement_percent)
    df["convergence_cutoff"] = df["improvement_percent"].apply(lambda values: int(np.argmax(values > 99.99)) if len(values) else 0)
    df["eval_limit"] = df["dim"].map(lambda dim: int(dim * eval_limit_factor))
    df["reduced_len"] = df.apply(lambda row: int((row["evals"] <= row["eval_limit"]).astype(int).sum()), axis=1)
    df["rank_len"] = df.apply(lambda row: int((row["rank_evals"] <= row["eval_limit"]).astype(int).sum()), axis=1)
    df["evals"] = df.apply(lambda row: row["evals"][: row["reduced_len"]], axis=1)
    df["vals"] = df.apply(lambda row: row["vals"][: row["reduced_len"]], axis=1)
    df["ranks"] = df.apply(lambda row: row["ranks"][: row["rank_len"]], axis=1)
    df = df[df["rank_len"] > 0].copy()
    df["last_rank"] = df.apply(lambda row: row["ranks"][-1], axis=1)
    df["rank_evals"] = df.apply(lambda row: row["rank_evals"][: row["rank_len"]], axis=1)
    if df.empty:
        return None

    grouped = df.groupby("model").agg({
        "last_rank": "mean",
        "elapsed_time": "mean",
        "model": "first",
        "model_kind": "first",
        "surrogate": "first",
        "gen_mult": "first",
    })
    ax = bar(grouped, y_name="last_rank", print_table=title, table_path=table_path, baselines=bundle.baselines, index_mapper=display_model_label)
    ax.set_title(title)
    ax.set_ylabel(RANK_LABEL)
    return save_and_show(graph_name, show=False, output_dir=output_dir)


def _init_eval_window_worker(data_dir: str | os.PathLike[str] | None) -> None:
    from control_analysis.data import load_control_bundle

    global _WORKER_BUNDLE
    _WORKER_BUNDLE = load_control_bundle(data_dir=data_dir)


def _render_eval_window_graph_worker(spec: EvalWindowGraphSpec, output_dir: str | os.PathLike[str]) -> Path | None:
    from control_analysis.data import load_control_bundle

    bundle = _WORKER_BUNDLE
    if bundle is None:
        bundle = load_control_bundle()
    return render_eval_window_graph(bundle=bundle, spec=spec, output_dir=output_dir)


def run_eval_window_graphs(
    data_dir: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str] = "graphs",
    max_workers: int | None = None,
) -> list[Path]:
    specs = _build_eval_window_graph_specs()
    if max_workers is not None and max_workers <= 1:
        from control_analysis.data import load_control_bundle

        bundle = load_control_bundle(data_dir=data_dir)
        return [
            path
            for path in (
                render_eval_window_graph(bundle=bundle, spec=spec, output_dir=output_dir)
                for spec in specs
            )
            if path is not None
        ]

    worker_count = max_workers or min(len(specs), os.cpu_count() or 1)
    if worker_count <= 1:
        from control_analysis.data import load_control_bundle

        bundle = load_control_bundle(data_dir=data_dir)
        return [
            path
            for path in (
                render_eval_window_graph(bundle=bundle, spec=spec, output_dir=output_dir)
                for spec in specs
            )
            if path is not None
        ]

    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_eval_window_worker,
        initargs=(data_dir,),
    ) as executor:
        results = executor.map(_render_eval_window_graph_worker, specs, repeat(output_dir))
        return [path for path in results if path is not None]


def plot_full_desc_ranking(bundle: ControlDataBundle, output_dir: str | os.PathLike[str] = "graphs") -> Path:
    grouped = default_groupby(bundle.df_og, "full_desc")
    _write_series_plot_table(
        grouped["avg_rank"],
        output_name="full_desc_ranking",
        output_dir=output_dir,
        index_label="Configuration",
        value_label="Average rank percentile",
    )
    ax = bar(grouped)
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(15)
    plt.xticks(size=5)
    return save_and_show("full_desc_ranking", show=False, output_dir=output_dir)




def plot_pure_population_size(bundle: ControlDataBundle, output_dir: str | os.PathLike[str] = "graphs") -> Path:
    grouped = default_groupby(bundle.pures, "pop_size")
    _write_series_plot_table(
        grouped["avg_rank"],
        output_name="pure_population_size",
        output_dir=output_dir,
        index_label="Population size",
        value_label="Average rank percentile",
        first_column_alignment="r",
    )
    ax = bar(grouped)
    ax.set_xlabel("Population Size")
    ax.set_title("Normal Evaluation")
    return save_and_show("pure_population_size", show=False, output_dir=output_dir)


def plot_gp_true_evaluations_by_population(bundle: ControlDataBundle, output_dir: str | os.PathLike[str] = "graphs") -> Path:
    df = bundle.df_og.copy()
    df = df[(df["model"] == "gp") & (df["dim_red_kind"] == "none")]
    df = default_groupby(df, ["true_evaluations", "pop_size"])
    pures2 = bundle.pures.set_index(bundle.pures["pop_size"].map(lambda value: (value, value)))
    df = pd.concat([df, pures2])
    export_df = df["avg_rank"].reset_index()
    export_df = export_df.rename(
        columns={
            "true_evaluations": "True evaluations",
            "pop_size": "Population size",
            "avg_rank": "Average rank percentile",
        }
    )
    _write_plot_table(export_df, output_name="gp_true_evaluations_by_population", output_dir=output_dir, first_column_alignment="r")
    bar(df, print_table=False)
    return save_and_show("gp_true_evaluations_by_population", show=False, output_dir=output_dir)




def plot_elapsed_time_by_dim_red_kind(bundle: ControlDataBundle, output_dir: str | os.PathLike[str] = "graphs") -> Path:
    grouped = default_groupby(bundle.df_og.copy(), "dim_red_kind")
    _write_series_plot_table(
        grouped["elapsed_time"],
        output_name="elapsed_time_by_dim_red_kind",
        output_dir=output_dir,
        index_label="Dimensionality reduction kind",
        value_label="Iteration time (ms)",
    )
    ax = bar(grouped, y_name="elapsed_time")
    ax.set_ylabel("Iteration Time (ms)")
    return save_and_show("elapsed_time_by_dim_red_kind", show=False, output_dir=output_dir)


def plot_model_comparison(bundle: ControlDataBundle, output_dir: str | os.PathLike[str] = "graphs") -> Path:
    df = bundle.df_og.copy()
    df = df[(df["dim_red_kind"] == "none") & (df["pop_size"] == 48) & (df["true_ratio"].map(Fraction) == Fraction(1, 8))]
    df = df[~df["model"].isin(SELECTOR_BASELINE_MODELS)].copy()
    grouped = default_groupby(df, "model")
    grouped.index = grouped.index.map(display_model_label)
    grouped = grouped.sort_index()
    _write_series_plot_table(
        grouped["avg_rank"],
        output_name="model_comparison",
        output_dir=output_dir,
        index_label="Model",
        value_label="Average rank percentile",
    )
    ax = bar(grouped)
    two_layer_tics(ax)
    return save_and_show("model_comparison", show=False, output_dir=output_dir)







NAMED_PLOT_JOBS = {
    "full_desc_ranking": plot_full_desc_ranking,
    "pure_population_size": plot_pure_population_size,
    "gp_true_evaluations_by_population": plot_gp_true_evaluations_by_population,
    "elapsed_time_by_dim_red_kind": plot_elapsed_time_by_dim_red_kind,
    "model_comparison": plot_model_comparison,
}


def run_named_plots(bundle: ControlDataBundle, names: list[str] | None = None, output_dir: str | os.PathLike[str] = "graphs") -> dict[str, str | None]:
    selected_names = names or list(NAMED_PLOT_JOBS.keys())
    results: dict[str, str | None] = {}
    for name in selected_names:
        path = NAMED_PLOT_JOBS[name](bundle=bundle, output_dir=output_dir)
        results[name] = None if path is None else str(path)
    return results


def run_stats_report(bundle: ControlDataBundle, output_dir: str | os.PathLike[str] = "graphs/avgs") -> dict[str, str | float | int | None]:
    report = write_stats_report(bundle.df_og, output_dir=output_dir)
    return {
        key: (str(value) if hasattr(value, "as_posix") else value)
        for key, value in report.items()
    }
