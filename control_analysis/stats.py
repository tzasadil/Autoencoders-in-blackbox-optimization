from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
import scikit_posthocs as sp
from statsmodels.stats.weightstats import ttost_paired

import pd_cols


ALPHA = 0.05
DEFAULT_EQUIVALENCE_DELTA = 0.5
DEFAULT_EQUIVALENCE_SWEEP = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.55, 3.57, 3.6, 4.0]
PRIMARY_DOE_MODEL = "doe_2_8"
MODEL_ORDER = [PRIMARY_DOE_MODEL, "doe_plain_2_8", "none", "gp", "nn3"]
MODEL_LABELS = {
    PRIMARY_DOE_MODEL: "DOE/VAE",
    "doe_plain_2_8": "DOE plain",
    "none": "No surrogate",
    "gp": "GP",
    "nn3": "NN",
}


def auc(df: pd.DataFrame) -> pd.Series:
    def auc_regret(x, y):
        return np.trapz(y, x)

    return df.groupby(pd_cols.determining_cols).apply(lambda group: auc_regret(group["evals"].to_numpy(), group["ranks"].to_numpy()))


def equivalence(df: pd.DataFrame, left_model: str = PRIMARY_DOE_MODEL, right_model: str = "none", delta: float = 0.5) -> dict[str, float]:
    wide = df.pivot_table(index=["function", "instance", "dim"], columns="model", values="avg_rank")
    left = wide[left_model].to_numpy()
    right = wide[right_model].to_numpy()
    p_value, _, _ = ttost_paired(left, right, low=-delta, upp=delta)
    return {"p_value": float(p_value), "delta": delta}


def stat_tests(df: pd.DataFrame, descriptor: str = "") -> dict[str, object]:
    wide = df.pivot_table(index=["function", "instance", "dim"], columns="model", values="avg_rank")
    algo_columns = list(wide.columns)
    friedman = scipy_stats.friedmanchisquare(*[wide[column].to_numpy() for column in algo_columns])
    posthoc = sp.posthoc_nemenyi_friedman(wide.to_numpy())
    posthoc.index = algo_columns
    posthoc.columns = algo_columns
    return {
        "descriptor": descriptor,
        "friedman_statistic": float(friedman.statistic),
        "friedman_pvalue": float(friedman.pvalue),
        "posthoc": posthoc,
    }


def prepare_model_kind_comparison(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df[df["model"].isin(MODEL_ORDER)]
        .groupby(["function", "instance", "dim", "model"], as_index=False)
        .agg(avg_rank=("avg_rank", "mean"))
    )
    return grouped


def compute_significance_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float, float, int]:
    wide = df.pivot_table(index=["function", "instance", "dim"], columns="model", values="avg_rank")
    ordered_columns = [column for column in MODEL_ORDER if column in wide.columns]
    wide = wide[ordered_columns].dropna()
    if len(wide.columns) < 2:
        raise RuntimeError("Need at least two model kinds for significance testing.")
    friedman = scipy_stats.friedmanchisquare(*[wide[column].to_numpy() for column in wide.columns])
    blocks = len(wide)
    kendalls_w = float(friedman.statistic) / (blocks * (len(wide.columns) - 1)) if blocks else math.nan

    nemenyi = sp.posthoc_nemenyi_friedman(wide.to_numpy())
    nemenyi.index = wide.columns
    nemenyi.columns = wide.columns

    rows: list[dict[str, object]] = []
    for model_name in wide.columns:
        p_value = np.nan if model_name == PRIMARY_DOE_MODEL else float(nemenyi.loc[model_name, PRIMARY_DOE_MODEL])
        rows.append(
            {
                "Model": MODEL_LABELS.get(model_name, model_name),
                "Nemenyi p vs DOE": p_value,
                "Significant vs DOE": "--" if np.isnan(p_value) else ("yes" if p_value < ALPHA else "no"),
            }
        )

    table = pd.DataFrame(rows)
    order_map = {MODEL_LABELS[key]: index for index, key in enumerate(MODEL_ORDER)}
    table["order"] = table["Model"].map(order_map)
    table = table.sort_values("order").drop(columns="order")
    return table, float(friedman.statistic), float(friedman.pvalue), kendalls_w, blocks


def compute_equivalence_test(
    df: pd.DataFrame,
    delta: float = DEFAULT_EQUIVALENCE_DELTA,
    alpha: float = ALPHA,
    left_model: str = PRIMARY_DOE_MODEL,
    right_model: str = "none",
) -> dict[str, float | int | bool | tuple[float, float]]:
    wide = df.pivot_table(index=["function", "instance", "dim"], columns="model", values="avg_rank")
    wide = wide[[column for column in MODEL_ORDER if column in wide.columns]].dropna()
    if left_model not in wide.columns or right_model not in wide.columns:
        raise RuntimeError(f"Missing model columns for equivalence test: {left_model}, {right_model}")

    left = wide[left_model].to_numpy()
    right = wide[right_model].to_numpy()
    diff = left - right
    p_tost, lower_result, upper_result = ttost_paired(left, right, low=-delta, upp=delta)
    confidence_interval = scipy_stats.t.interval(
        1 - 2 * alpha,
        len(diff) - 1,
        loc=float(np.mean(diff)),
        scale=scipy_stats.sem(diff),
    )

    return {
        "left_model": left_model,
        "right_model": right_model,
        "delta": delta,
        "alpha": alpha,
        "blocks": len(diff),
        "mean_difference": float(np.mean(diff)),
        "p_lower": float(lower_result[1]),
        "p_upper": float(upper_result[1]),
        "p_tost": float(p_tost),
        "equivalent": bool(p_tost < alpha),
        "ci_90": tuple(float(bound) for bound in confidence_interval),
    }


def sweep_equivalence_margins(
    df: pd.DataFrame,
    margins: list[float] | None = None,
    alpha: float = ALPHA,
    left_model: str = PRIMARY_DOE_MODEL,
    right_model: str = "none",
) -> list[dict[str, float | int | bool | tuple[float, float]]]:
    selected_margins = margins or DEFAULT_EQUIVALENCE_SWEEP
    return [
        compute_equivalence_test(df, delta=margin, alpha=alpha, left_model=left_model, right_model=right_model)
        for margin in selected_margins
    ]


def find_first_equivalent_margin(sweep_results: list[dict[str, float | int | bool | tuple[float, float]]]) -> float | None:
    for result in sweep_results:
        if bool(result["equivalent"]):
            return float(result["delta"])
    return None


def export_significance_table(
    table: pd.DataFrame,
    output_path: str | os.PathLike[str],
    friedman_stat: float,
    friedman_p: float,
    kendalls_w: float,
    blocks: int,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line_break = r"\\"

    def format_p_value(value: float) -> str:
        if np.isnan(value):
            return "--"
        return f"{value:.6g}"

    lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        f"Model & Nemenyi $p$ vs DOE & Significant vs DOE {line_break}",
        "\\midrule",
    ]
    for row in table.to_dict(orient="records"):
        lines.append(
            f"{row['Model']} & {format_p_value(row['Nemenyi p vs DOE'])} & {row['Significant vs DOE']} {line_break}"
        )
    lines.extend(
        [
            "\\midrule",
            f"\\multicolumn{{3}}{{l}}{{Friedman $\\chi^2={friedman_stat:.2f}$, $p = {friedman_p:.6g}$, Kendall's $W={kendalls_w:.3f}$, $N={blocks}$ blocks.}} {line_break}",
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_stats_report(
    df: pd.DataFrame,
    output_dir: str | os.PathLike[str] = "graphs/avgs",
    equivalence_delta: float = DEFAULT_EQUIVALENCE_DELTA,
    equivalence_sweep: list[float] | None = None,
) -> dict[str, object]:
    comparison_df = prepare_model_kind_comparison(df)
    table, friedman_stat, friedman_p, kendalls_w, blocks = compute_significance_summary(comparison_df)
    equivalence = compute_equivalence_test(comparison_df, delta=equivalence_delta)
    sweep_results = sweep_equivalence_margins(comparison_df, margins=equivalence_sweep)
    first_equivalent_margin = find_first_equivalent_margin(sweep_results)

    output_path = Path(output_dir)
    latex_path = export_significance_table(
        table=table,
        output_path=output_path / "stat_significance_summary.tex",
        friedman_stat=friedman_stat,
        friedman_p=friedman_p,
        kendalls_w=kendalls_w,
        blocks=blocks,
    )
    json_path = output_path / "stat_significance_summary.json"
    payload = {
        "friedman_statistic": friedman_stat,
        "friedman_pvalue": friedman_p,
        "kendalls_w": kendalls_w,
        "blocks": blocks,
        "table": table.to_dict(orient="records"),
        "equivalence": equivalence,
        "equivalence_sweep": sweep_results,
        "first_equivalent_margin": first_equivalent_margin,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "latex_path": latex_path,
        "json_path": json_path,
        "comparison_rows": len(comparison_df),
        "friedman_statistic": friedman_stat,
        "friedman_pvalue": friedman_p,
        "first_equivalent_margin": first_equivalent_margin,
    }