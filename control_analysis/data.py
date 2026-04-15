from __future__ import annotations

import math
import os
from fractions import Fraction

import numpy as np
import pandas as pd

import pd_cols
import storage
from control_analysis.models import ControlDataBundle, EvalWindowGraphSpec
from control_analysis.transforms import default_groupby, derive_dim_red_kind


def _to_rank_percentiles(values: pd.Series, eval_limit: int | None = None) -> list[np.ndarray]:
    arrays = values.to_list()
    results = [[] for _ in range(len(arrays))]
    max_len = max(map(len, arrays))
    if eval_limit is not None:
        max_len = min(max_len, eval_limit)
    for index in range(max_len):
        column = [(array[index] if index < len(array) else None) for array in arrays]
        mask = [item is not None for item in column]
        nonempty = np.array([item for item in column if item is not None])
        if len(nonempty) == 0:
            break
        if len(nonempty) == 1:
            only_index = np.argmax(mask)
            results[only_index].append(100.0)
            continue
        bigger = (nonempty[:, None] > nonempty[None, :]).sum(axis=1)
        equal = (nonempty[:, None] == nonempty[None, :]).sum(axis=1) - 1
        percentiles = 100 - (bigger + (equal / 2.0)) / len(nonempty) * 100
        percentile_index = 0
        for result_index, is_present in enumerate(mask):
            if is_present:
                results[result_index].append(percentiles[percentile_index])
                percentile_index += 1
    return [np.array(result) for result in results]


def compute_control_ranks(df: pd.DataFrame, eval_limit: int | None = None) -> pd.DataFrame:
    ranked = df.copy()

    def normalise_row(row: pd.Series) -> np.ndarray:
        evals = row["evals"]
        vals = row["vals"]
        dim = row["dim"]
        suggested_population = dim * 5
        if eval_limit is not None:
            evals = evals[:eval_limit]
            vals = vals[:eval_limit]
        if suggested_population == row["pop_size"]:
            return vals
        sample_points = np.array(range(suggested_population, 250 * dim + 1, suggested_population))
        current_index = np.argmin(evals >= suggested_population)
        evals = evals[current_index:]
        vals = vals[current_index:]
        sampled_values = []
        for sample_point in sample_points:
            while current_index < len(evals) and sample_point > evals[current_index]:
                current_index += 1
            if current_index >= len(evals):
                sampled_values.append(vals[-1])
            else:
                sampled_values.append(vals[current_index])
        return np.array(sampled_values)

    ranked["normalised_len_vals"] = ranked.apply(normalise_row, axis=1)
    ranked["ranks"] = ranked.groupby(["function", "instance", "dim"])["normalised_len_vals"].transform(
        lambda values: _to_rank_percentiles(values, eval_limit=eval_limit)
    )
    ranked = ranked.drop(["normalised_len_vals"], axis=1)
    ranked["avg_rank"] = ranked["ranks"].apply(np.mean)
    ranked["last_rank"] = ranked["ranks"].apply(lambda values: values[-1])
    ranked["median_rank"] = ranked["ranks"].apply(np.median)
    return ranked


def df_enhance(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["dim_red"] = enriched["dim_red"].replace("", "none")
    enriched["gen_mult"] = enriched["gen_mult"].map(int)
    enriched["model"] = enriched["model"].replace("", "none")
    enriched["pop_size"] = enriched["pop_size"].map(
        lambda value: int(value) if str(value).isdigit() else str(value).replace("None", "none")
    )
    enriched["actual_pop_size"] = enriched.apply(
        lambda row: row["pop_size"] if row["pop_size"] != "none" else 4 + math.floor(3 * math.log(row["dim"])),
        axis=1,
    )
    enriched["rank_evals"] = enriched.apply(
        lambda row: np.array(range(5 * row["dim"], 250 * row["dim"] + 1, 5 * row["dim"])),
        axis=1,
    )
    enriched["model_kind"] = enriched["model"].map(str)
    enriched["surrogate"] = enriched["model"].map(str)
    enriched["dim_red_kind"] = enriched["dim_red"].map(derive_dim_red_kind)
    enriched["true_ratio"] = enriched["gen_mult"].map(lambda value: 1.0 / float(value) if value else np.nan)
    enriched["true_evaluations"] = enriched.apply(
        lambda row: int(row["actual_pop_size"] * row["true_ratio"]) if row["actual_pop_size"] != "none" else np.nan,
        axis=1,
    )
    enriched["scale_train"] = False
    return enriched


def load_control_bundle(data_dir: str | os.PathLike[str] | None = None) -> ControlDataBundle:
    df_og = storage.merge_and_load(data_dir=data_dir)
    if df_og is None or df_og.empty:
        raise RuntimeError("No control notebook data found.")
    df_og = df_og[(df_og["note"] == "") | (df_og["note"] == "none")].copy()
    df_og["full_desc"] = df_og.apply(pd_cols.get_full_desc, axis=1)
    df_og = df_enhance(df_og)
    df_og = compute_control_ranks(df_og)
    pure_mask = df_og["gen_mult"].map(int) == 1
    pures = df_og[pure_mask].copy()
    pca_mask = (
        (df_og["model"] == "gp")
        & (df_og["dim_red_kind"] == "pca")
        & (df_og["pop_size"].isin([48, 64]))
        & (df_og["true_ratio"].map(Fraction) == Fraction(1, 8))
    )
    pca_df = df_og[pca_mask].copy()
    baselines = default_groupby(pures, ["pop_size"])
    return ControlDataBundle(df_og=df_og, pures=pures, baselines=baselines, pca_df=pca_df)