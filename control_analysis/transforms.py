from __future__ import annotations

from collections.abc import Callable, Sequence
import numpy as np
import pandas as pd

from control_analysis.constants import FUNC_GROUP_LABELS


def p(func, /, *args, **keywords):
    def newfunc(*fargs, **fkeywords):
        newkeywords = {**keywords, **fkeywords}
        return func(*args, *fargs, **newkeywords)

    newfunc.func = func
    newfunc.args = args
    newfunc.keywords = keywords
    return newfunc


def np_apply_axis0(fn: Callable[[np.ndarray], float] | None = None) -> Callable[[pd.Series], list[float]]:
    def inner(values: pd.Series) -> list[float]:
        arrays = values.to_list()
        return list(np.apply_along_axis(fn, 0, arrays))

    return inner


def avg_axis0_rugged(values: pd.Series) -> list[float]:
    arrays = values.to_list()
    max_len = max(map(len, arrays))
    masks = []
    padded = []
    for array in arrays:
        to_pad = max_len - len(array)
        masks.append([True] * len(array) + [False] * to_pad)
        padded.append(np.pad(array, (0, to_pad), constant_values=0))
    stacked = np.stack(padded, axis=0)
    num_valid = np.sum(np.array(masks), axis=0)
    averages = np.sum(stacked, axis=0) / num_valid
    return list(averages)


def close_to(series: pd.Series, number: float) -> pd.Series:
    return series.map(lambda value: abs(value - number) <= 1e-3)


def get_param_desc_title(df: pd.DataFrame) -> str:
    def title_stringer(prefix: str, column_name: str) -> str:
        minimum = df[column_name].min()
        maximum = df[column_name].max()
        suffix = f"-{maximum}" if minimum != maximum else ""
        return f"{prefix} {minimum}{suffix}"

    dims = ", ".join(str(value) for value in np.unique(df["dim"]))
    return f"{title_stringer('fun', 'function')}; dim {dims}{title_stringer('; inst', 'instance')}"


def default_groupby(df: pd.DataFrame, columns: str | Sequence[str]) -> pd.DataFrame:
    grouping_columns = [columns] if isinstance(columns, str) else list(columns)
    aggregation_map: dict[str, str | Callable[[pd.Series], list[float]]] = {
        "ranks": np_apply_axis0(np.average),
        "avg_rank": "mean",
        "last_rank": "mean",
        "elapsed_time": "mean",
        "model": "first",
        "model_kind": "first",
        "surrogate": "first",
        "gen_mult": "first",
    }
    for column in grouping_columns:
        aggregation_map.pop(column, None)
    available_map = {column: agg for column, agg in aggregation_map.items() if column in df.columns}
    return df.groupby(grouping_columns).agg(available_map)


def add_func_group(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["func_group"] = "Other"
    enriched["func_group_key"] = "other"
    for key, start, end, label in FUNC_GROUP_LABELS:
        mask = enriched["function"].between(start, end)
        enriched.loc[mask, "func_group"] = label
        enriched.loc[mask, "func_group_key"] = key
    return enriched


def derive_dim_red_kind(dim_red_name: str) -> str:
    dim_red_name = str(dim_red_name)
    prefix = []
    for char in dim_red_name:
        if not char.isalpha():
            break
        prefix.append(char)
    return "".join(prefix).lower() or "none"


def improvement_percent(values) -> np.ndarray:
    array = np.asarray(values)
    if len(array) < 4:
        return np.zeros_like(array, dtype=float)
    denominator = array[3] - array[-1]
    if abs(denominator) < 1e-12:
        return np.zeros_like(array, dtype=float)
    return 100 * (1 - (array - array[-1]) / denominator)