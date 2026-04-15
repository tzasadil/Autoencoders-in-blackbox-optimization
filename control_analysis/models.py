from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ControlDataBundle:
    df_og: pd.DataFrame
    pures: pd.DataFrame
    baselines: pd.DataFrame
    pca_df: pd.DataFrame


@dataclass(frozen=True)
class EvalWindowGraphSpec:
    frac_eval_limit: int
    dim: int | None
    func_start: int
    func_end: int
    description: str