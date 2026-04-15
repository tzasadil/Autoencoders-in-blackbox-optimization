from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from control_analysis.constants import TABLE_DEFAULT_COLUMN_FORMAT


def latex_tabular(series: pd.Series) -> str:
    def label(value: object) -> str:
        return " " if value is None else str(value).replace("_", " ")

    table = [f"\\begin{{tabular}}{{{TABLE_DEFAULT_COLUMN_FORMAT}}}", "\\hline"]
    table.append(f"{label(series.index.name)} & {label(series.name)}\\\\")
    table.append("\\hline")
    for key, value in series.items():
        table.append(f"{key} & {value:0.2f} \\\\")
    table.append("\\hline")
    table.append("\\end{tabular}")
    return "\n".join(table)


def write_latex_table(series: pd.Series, output_path: str | os.PathLike[str], heading: str | None = None) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if heading:
            handle.write(f"% {heading}\n")
        handle.write(latex_tabular(series))
        handle.write("\n")
    return path


def print_latex(series: pd.Series, output_path: str | os.PathLike[str] | None = None, heading: str | None = None) -> None:
    table = latex_tabular(series)
    if output_path is None:
        print(table)
        return
    write_latex_table(series, output_path=output_path, heading=heading)


def write_dataframe_tabular(df: pd.DataFrame, output_path: Path, column_format: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latex = df.to_latex(index=False, escape=False, float_format=lambda value: f"{value:0.2f}", column_format=column_format)
    output_path.write_text("\n".join(latex.splitlines()) + "\n", encoding="utf-8")
    return output_path