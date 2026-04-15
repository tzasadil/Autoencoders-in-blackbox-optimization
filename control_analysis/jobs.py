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

from control_analysis.constants import EVAL_WINDOW_FUNC_GROUPS, FUNC_GROUP_LABELS
from control_analysis.formatting import write_dataframe_tabular
from control_analysis.models import ControlDataBundle, EvalWindowGraphSpec
from control_analysis.plotting import bar, save_and_show, two_layer_tics
from control_analysis.stats import write_stats_report
from control_analysis.transforms import add_func_group, default_groupby, improvement_percent


_WORKER_BUNDLE: ControlDataBundle | None = None
PRIMARY_DOE_MODEL = "doe_2_8"
COMPARISON_MODELS = ["none", "gp", "nn3", "doe_plain_2_8", PRIMARY_DOE_MODEL]
_MODEL_DISPLAY_ORDER = COMPARISON_MODELS


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


def _plot_metric_bar(
    summary: pd.DataFrame,
    value_column: str,
    title: str,
    ylabel: str,
    output_name: str,
    output_dir: str | os.PathLike[str],
) -> Path | None:
    if summary.empty:
        return None
    plotting_df = summary[[value_column]].copy()
    plotting_df = plotting_df.dropna(subset=[value_column])
    if plotting_df.empty:
        return None
    ax = bar(plotting_df, y_name=value_column)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return save_and_show(output_name, show=False, output_dir=output_dir)


def _plot_generation_progress(
    summary: pd.DataFrame,
    title: str,
    ylabel: str,
    output_name: str,
    output_dir: str | os.PathLike[str],
) -> Path | None:
    if summary.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(summary["generation_fraction"], summary["avg_spearman_corr"], color="forestgreen", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Fraction of generations")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    return save_and_show(output_name, show=False, output_dir=output_dir)


def _build_problem_metric_summary(df_og: pd.DataFrame) -> pd.DataFrame:
    analysis_df = add_func_group(df_og)
    analysis_df["avg_rank"] = analysis_df["ranks"].apply(np.mean)
    analysis_df["avg_spearman_corr"] = analysis_df["spearman_corr"].apply(_nanmean_array)
    analysis_df["avg_spearman_pval"] = analysis_df["spearman_pval"].apply(_nanmean_array)

    return analysis_df.groupby(
        ["func_group", "func_group_key", "function", "instance", "dim", "model", "true_ratio"],
        as_index=False,
    ).agg(
        avg_rank=("avg_rank", "mean"),
        avg_spearman_corr=("avg_spearman_corr", "mean"),
        avg_spearman_pval=("avg_spearman_pval", "mean"),
        elapsed_time=("elapsed_time", "sum"),
    )


def _build_generation_fraction_summary(df_og: pd.DataFrame) -> pd.DataFrame:
    doe_rows = df_og[df_og["model"] == PRIMARY_DOE_MODEL]
    if doe_rows.empty:
        return pd.DataFrame(columns=["generation_fraction", "avg_spearman_corr"])

    arrays = doe_rows["spearman_corr"].map(lambda values: np.asarray(values, dtype=float)).tolist()
    if not arrays:
        return pd.DataFrame(columns=["generation_fraction", "avg_spearman_corr"])

    stacked = np.vstack(arrays)
    num_generations = stacked.shape[1]
    return pd.DataFrame(
        {
            "generation_fraction": np.arange(1, num_generations + 1, dtype=float) / num_generations,
            "avg_spearman_corr": np.nanmean(stacked, axis=0),
        }
    )


def _render_model_breakdown_graphs(per_problem: pd.DataFrame, output_dir: str | os.PathLike[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    metric_specs = [
        ("avg_spearman_corr", "surr_correlation", "Average Spearman correlation", "Surrogate correlation", True),
        ("avg_spearman_pval", "surr_pval", "Average Spearman p-value", "Surrogate p-value", True),
        ("avg_rank", "avg_rank", "Average rank percentile", "Average rank", False),
    ]

    for dim in [2, 5, 10, None]:
        dim_label = "all_dims" if dim is None else f"dim_{dim}"
        title_suffix = "all dims average" if dim is None else f"dim={dim}"
        filtered = per_problem if dim is None else per_problem[per_problem["dim"] == dim]
        if filtered.empty:
            continue
        model_summary = filtered.groupby("model", as_index=False).agg(
            avg_spearman_corr=("avg_spearman_corr", "mean"),
            avg_spearman_pval=("avg_spearman_pval", "mean"),
            avg_rank=("avg_rank", "mean"),
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
        .agg(avg_rank=("avg_rank", "mean"), avg_spearman_corr=("avg_spearman_corr", "mean"))
        .set_index("func_group")
        .loc[ordered_groups]
    )
    doe_by_dim = (
        doe_summary.groupby("dim", as_index=False)
        .agg(avg_rank=("avg_rank", "mean"), avg_spearman_corr=("avg_spearman_corr", "mean"))
        .sort_values("dim")
        .set_index("dim")
    )
    doe_by_fraction = _build_generation_fraction_summary(df_og)

    bar_specs = [
        (doe_by_func_group, "avg_rank", "DOE average rank by function group", "Average rank percentile", "doe_avg_rank_by_func_group"),
        (doe_by_dim, "avg_rank", "DOE average rank by dimension", "Average rank percentile", "doe_avg_rank_by_dim"),
        (doe_by_func_group, "avg_spearman_corr", "DOE surrogate correlation by function group", "Average Spearman correlation", "doe_spearman_by_func_group"),
        (doe_by_dim, "avg_spearman_corr", "DOE surrogate correlation by dimension", "Average Spearman correlation", "doe_spearman_by_dim"),
    ]

    for summary, value_column, title, ylabel, output_name in bar_specs:
        path = _plot_metric_bar(summary, value_column=value_column, title=title, ylabel=ylabel, output_name=output_name, output_dir=output_dir)
        if path is not None:
            paths[output_name] = path

    generation_path = _plot_generation_progress(
        doe_by_fraction,
        title="DOE surrogate correlation by generation progress",
        ylabel="Average Spearman correlation",
        output_name="doe_spearman_by_generation_fraction",
        output_dir=output_dir,
    )
    if generation_path is not None:
        paths["doe_spearman_by_generation_fraction"] = generation_path

    return paths


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
    )


def run_doe_group_analysis(df_og: pd.DataFrame, output_dir: str | os.PathLike[str] = "graphs/avgs") -> dict[str, Path | pd.DataFrame]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    analysis_df = add_func_group(df_og)
    analysis_df = analysis_df[analysis_df["model"].isin(COMPARISON_MODELS)].copy()
    analysis_df["avg_rank"] = analysis_df["ranks"].apply(np.mean)
    per_problem_metrics = _build_problem_metric_summary(analysis_df)

    per_problem = analysis_df.groupby(
        ["func_group", "func_group_key", "dim", "function", "instance", "model"],
        as_index=False,
    ).agg(avg_rank=("avg_rank", "mean"))

    group_summary = per_problem.groupby(["func_group", "func_group_key", "dim", "model"], as_index=False).agg(
        avg_rank=("avg_rank", "mean"),
        problems=("avg_rank", "size"),
    )

    doe_summary = group_summary[group_summary["model"] == PRIMARY_DOE_MODEL].rename(columns={"avg_rank": "doe_avg_rank"})
    baseline_summary = group_summary[group_summary["model"] == "none"].rename(columns={"avg_rank": "baseline_avg_rank"})
    peer_summary = (
        group_summary[group_summary["model"] != PRIMARY_DOE_MODEL]
        .groupby(["func_group", "func_group_key", "dim"], as_index=False)
        .agg(peer_avg_rank=("avg_rank", "mean"), best_non_doe_avg_rank=("avg_rank", "min"))
    )

    doe_group_eval = doe_summary.merge(
        baseline_summary[["func_group", "dim", "baseline_avg_rank"]],
        on=["func_group", "dim"],
        how="left",
    ).merge(peer_summary, on=["func_group", "func_group_key", "dim"], how="left")

    rank_within_group = group_summary.copy()
    rank_within_group["model_rank"] = rank_within_group.groupby(["func_group", "dim"])["avg_rank"].rank(method="dense")
    doe_rank = rank_within_group[rank_within_group["model"] == PRIMARY_DOE_MODEL][["func_group", "dim", "model_rank"]]
    doe_group_eval = doe_group_eval.merge(doe_rank, on=["func_group", "dim"], how="left")

    doe_group_eval["vs_baseline"] = doe_group_eval["baseline_avg_rank"] - doe_group_eval["doe_avg_rank"]
    doe_group_eval["vs_peer_mean"] = doe_group_eval["peer_avg_rank"] - doe_group_eval["doe_avg_rank"]
    doe_group_eval["vs_best_non_doe"] = doe_group_eval["best_non_doe_avg_rank"] - doe_group_eval["doe_avg_rank"]
    doe_group_eval = doe_group_eval.sort_values(["dim", "func_group_key"])

    doe_by_dim = doe_group_eval.groupby("dim", as_index=False).agg(
        doe_avg_rank=("doe_avg_rank", "mean"),
        vs_baseline=("vs_baseline", "mean"),
        vs_peer_mean=("vs_peer_mean", "mean"),
        vs_best_non_doe=("vs_best_non_doe", "mean"),
        mean_rank=("model_rank", "mean"),
    ).sort_values("dim")

    doe_by_func_group = doe_group_eval.groupby(["func_group", "func_group_key"], as_index=False).agg(
        doe_avg_rank=("doe_avg_rank", "mean"),
        vs_baseline=("vs_baseline", "mean"),
        vs_peer_mean=("vs_peer_mean", "mean"),
        vs_best_non_doe=("vs_best_non_doe", "mean"),
        mean_rank=("model_rank", "mean"),
    ).sort_values("func_group_key")

    doe_by_dim_export = doe_by_dim.rename(
        columns={"dim": "Dimension", "doe_avg_rank": "DOE avg. rank", "vs_baseline": "DOE - baseline", "mean_rank": "DOE rank"}
    )[["Dimension", "DOE avg. rank", "DOE - baseline", "DOE rank"]]
    doe_by_func_group_export = doe_by_func_group.rename(
        columns={"func_group": "Function group", "doe_avg_rank": "DOE avg. rank", "vs_baseline": "DOE - baseline", "mean_rank": "DOE rank"}
    )[["Function group", "DOE avg. rank", "DOE - baseline", "DOE rank"]]

    dim_table_path = write_dataframe_tabular(doe_by_dim_export, output_path / "doe_by_dim_summary.tex", "rccc")
    group_table_path = write_dataframe_tabular(doe_by_func_group_export, output_path / "doe_by_func_group_summary.tex", "lccc")

    heatmap_path = output_path / "doe_group_heatmaps.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    ordered_labels = [label for _, _, _, label in FUNC_GROUP_LABELS]
    sns.heatmap(
        doe_group_eval.pivot(index="func_group", columns="dim", values="vs_baseline").loc[ordered_labels],
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        ax=axes[0],
    )
    axes[0].set_title("DOE vs baseline none\npositive = DOE better")
    axes[0].set_xlabel("Dimension")
    axes[0].set_ylabel("Function group")
    sns.heatmap(
        doe_group_eval.pivot(index="func_group", columns="dim", values="model_rank").loc[ordered_labels],
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

    model_breakdown_paths = _render_model_breakdown_graphs(per_problem_metrics, output_dir=output_path)
    doe_focus_paths = _render_doe_focus_graphs(per_problem_metrics, analysis_df, output_dir=output_path)
    runtime_path = _render_runtime_graph(analysis_df, output_dir=output_path)

    result: dict[str, Path | pd.DataFrame] = {
        "dim_table_path": dim_table_path,
        "group_table_path": group_table_path,
        "heatmap_path": heatmap_path,
        "doe_by_dim": doe_by_dim_export,
        "doe_by_func_group": doe_by_func_group_export,
    }
    result.update({key: value for key, value in model_breakdown_paths.items()})
    result.update({key: value for key, value in doe_focus_paths.items()})
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
    title_parts = []
    eval_limit = 999
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
        eval_limit = int(250 / spec.frac_eval_limit)
    if df.empty:
        return None

    title = ", ".join(title_parts)
    graph_name = title.replace("/", "")
    table_path = Path(output_dir) / f"{graph_name}.tex"

    df["improvement_percent"] = df["vals"].apply(improvement_percent)
    df["convergence_cutoff"] = df["improvement_percent"].apply(lambda values: int(np.argmax(values > 99.99)) if len(values) else 0)
    df["reduced_len"] = df.apply(lambda row: int((row["evals"] <= eval_limit).astype(int).sum()), axis=1)
    df["rank_len"] = df.apply(lambda row: int((row["rank_evals"] <= eval_limit).astype(int).sum()), axis=1)
    df["evals"] = df.apply(lambda row: row["evals"][: row["reduced_len"]], axis=1)
    df["vals"] = df.apply(lambda row: row["vals"][: row["reduced_len"]], axis=1)
    df["ranks"] = df.apply(lambda row: row["ranks"][: row["rank_len"]], axis=1)
    df["avg_rank"] = df.apply(lambda row: row["ranks"].mean(), axis=1)
    df["last_rank"] = df.apply(lambda row: row["ranks"][-1], axis=1)
    df["rank_evals"] = df.apply(lambda row: row["rank_evals"][: row["rank_len"]], axis=1)
    if df.empty:
        return None

    grouped = df.groupby("model").agg({
        "avg_rank": "mean",
        "last_rank": "mean",
        "elapsed_time": "mean",
        "model": "first",
        "model_kind": "first",
        "surrogate": "first",
        "gen_mult": "first",
    })
    ax = bar(grouped, y_name="avg_rank", print_table=title, table_path=table_path, baselines=bundle.baselines)
    ax.set_title(title)
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


def run_eval_window_graphs(data_dir: str | os.PathLike[str] | None = None, output_dir: str | os.PathLike[str] = "graphs", max_workers: int | None = None) -> list[Path]:
    specs = _build_eval_window_graph_specs()
    if max_workers is not None and max_workers <= 1:
        from control_analysis.data import load_control_bundle

        bundle = load_control_bundle(data_dir=data_dir)
        return [path for path in (render_eval_window_graph(bundle=bundle, spec=spec, output_dir=output_dir) for spec in specs) if path is not None]

    worker_count = max_workers or min(len(specs), os.cpu_count() or 1)
    if worker_count <= 1:
        from control_analysis.data import load_control_bundle

        bundle = load_control_bundle(data_dir=data_dir)
        return [path for path in (render_eval_window_graph(bundle=bundle, spec=spec, output_dir=output_dir) for spec in specs) if path is not None]

    with ProcessPoolExecutor(max_workers=worker_count, initializer=_init_eval_window_worker, initargs=(data_dir,)) as executor:
        results = executor.map(_render_eval_window_graph_worker, specs, repeat(output_dir))
        return [path for path in results if path is not None]


def plot_full_desc_ranking(bundle: ControlDataBundle, output_dir: str | os.PathLike[str] = "graphs") -> Path:
    ax = bar(bundle.df_og, "full_desc")
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(15)
    plt.xticks(size=5)
    return save_and_show("full_desc_ranking", show=False, output_dir=output_dir)




def plot_pure_population_size(bundle: ControlDataBundle, output_dir: str | os.PathLike[str] = "graphs") -> Path:
    ax = bar(bundle.pures, "pop_size")
    ax.set_xlabel("Population Size")
    ax.set_title("Normal Evaluation")
    return save_and_show("pure_population_size", show=False, output_dir=output_dir)


def plot_pca_ratio_gp(bundle: ControlDataBundle, output_dir: str | os.PathLike[str] = "graphs") -> Path | None:
    if bundle.pca_df.empty:
        return None
    df = bundle.pca_df[bundle.pca_df["scale_train"] == False].copy()
    df = df[(df["model"] == "gp") & (df["pop_size"] == 64)]
    if df.empty:
        return None
    df["pca_ratio"] = df["dim_red"].map(lambda value: str(value).removeprefix("pca"))
    df1 = bundle.df_og[
        (bundle.df_og["model"] == "gp")
        & (bundle.df_og["dim_red_kind"] == "none")
        & (bundle.df_og["pop_size"] == 64)
        & (bundle.df_og["true_ratio"].map(Fraction) == Fraction(1, 8))
    ].iloc[0].copy()
    df1["pca_ratio"] = str(1.0)
    df.loc[str(len(df))] = df1
    ax = bar(df, "pca_ratio", regr=True, baseline_i=8, baselines=bundle.baselines)
    ax.set_xlabel("pca reduction ratio")
    ax.set_title("PCA + GP")
    return save_and_show("pca_ratio_gp", show=False, output_dir=output_dir)


def plot_gp_true_evaluations_by_population(bundle: ControlDataBundle, output_dir: str | os.PathLike[str] = "graphs") -> Path:
    df = bundle.df_og.copy()
    df = df[(df["model"] == "gp") & (df["dim_red_kind"] == "none")]
    df = default_groupby(df, ["true_evaluations", "pop_size"])
    pures2 = bundle.pures.set_index(bundle.pures["pop_size"].map(lambda value: (value, value)))
    df = pd.concat([df, pures2])
    bar(df, print_table=False)
    return save_and_show("gp_true_evaluations_by_population", show=False, output_dir=output_dir)


def plot_gp_dim_reduction(bundle: ControlDataBundle, output_dir: str | os.PathLike[str] = "graphs") -> Path:
    df = bundle.df_og.copy()
    df = df[df["model"] == "gp"]
    bar(df, "dim_red")
    return save_and_show("gp_dim_reduction", show=False, output_dir=output_dir)


def plot_elapsed_time_by_dim_red_kind(bundle: ControlDataBundle, output_dir: str | os.PathLike[str] = "graphs") -> Path:
    ax = bar(bundle.df_og.copy(), "dim_red_kind", "elapsed_time")
    ax.set_ylabel("Iteration Time (ms)")
    return save_and_show("elapsed_time_by_dim_red_kind", show=False, output_dir=output_dir)


def plot_model_comparison(bundle: ControlDataBundle, output_dir: str | os.PathLike[str] = "graphs") -> Path:
    df = bundle.df_og.copy()
    df = df[(df["dim_red_kind"] == "none") & (df["pop_size"] == 48) & (df["true_ratio"].map(Fraction) == Fraction(1, 8))]
    ax = bar(df, "model")
    two_layer_tics(ax)
    return save_and_show("model_comparison", show=False, output_dir=output_dir)


def plot_dim_red_kind_ranking(bundle: ControlDataBundle, output_dir: str | os.PathLike[str] = "graphs") -> Path:
    bar(bundle.df_og.copy(), "dim_red_kind")
    return save_and_show("dim_red_kind_ranking", show=False, output_dir=output_dir)


def plot_gp_true_ratio(bundle: ControlDataBundle, output_dir: str | os.PathLike[str] = "graphs") -> Path:
    df = bundle.df_og.copy()
    df = df[(df["model"] == "gp") & (df["dim_red_kind"] == "none")]
    ax = bar(df, "true_ratio", index_mapper=lambda value: Fraction(value))
    ax.set_label("dim red, true evals, aux evals")
    ax.set_xlabel("truly evaluated fraction of population")
    ax.set_ylabel("rank percentile avg")
    labels = [item.get_text() for item in ax.get_xticklabels()]
    if len(labels) > 1:
        labels[1] = "1/12"
        ax.set_xticklabels(labels)
    return save_and_show("gp_true_ratio", show=False, output_dir=output_dir)


NAMED_PLOT_JOBS = {
    "full_desc_ranking": plot_full_desc_ranking,
    "true_ratio_ranking": plot_true_ratio_ranking,
    "pure_population_size": plot_pure_population_size,
    "pca_ratio_gp": plot_pca_ratio_gp,
    "gp_true_evaluations_by_population": plot_gp_true_evaluations_by_population,
    "gp_dim_reduction": plot_gp_dim_reduction,
    "elapsed_time_by_dim_red_kind": plot_elapsed_time_by_dim_red_kind,
    "model_comparison": plot_model_comparison,
    "dim_red_kind_ranking": plot_dim_red_kind_ranking,
    "gp_true_ratio": plot_gp_true_ratio,
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