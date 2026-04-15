from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LATEXDIFF_ROOT = ROOT.parent / "latexdiff"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    main_py = read_text(ROOT / "main.py")
    config = json.loads(read_text(ROOT / "data" / "doe_sweep" / "best_doe_config.json"))
    results_tex = read_text(LATEXDIFF_ROOT / "new" / "en" / "pages" / "results.tex")
    implementation_tex = read_text(LATEXDIFF_ROOT / "new" / "en" / "pages" / "implementation.tex")

    require(
        re.search(r'DEFAULT_PROBLEM_INFO\s*=\s*"function_indices:1-24 dimensions:2,5,10 instance_indices:1-10"', main_py)
        is not None,
        "main.py no longer matches the thesis claim that the main comparison uses dimensions 2,5,10 and instances 1-10.",
    )

    require(config.get("model") == "doe_2_8", "best_doe_config.json no longer selects doe_2_8.")
    require(config.get("n_samples") == 2, "best_doe_config.json no longer records n_samples=2.")
    require(config.get("latent_dim") == 8, "best_doe_config.json no longer records latent_dim=8.")

    require(
        "Higher DOE avg. rank is better" in results_tex and "lower DOE rank is better" in results_tex,
        "results.tex no longer explains the direction of DOE avg. rank and DOE rank consistently.",
    )
    require(
        "\\texttt{doe\\_2\\_8}" in results_tex,
        "results.tex no longer names the selected DOE configuration as doe_2_8.",
    )
    require(
        "instances 1-10" in implementation_tex,
        "implementation.tex no longer matches the configured instance range 1-10.",
    )

    print("Thesis export consistency checks passed.")


if __name__ == "__main__":
    main()