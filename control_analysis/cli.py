from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")

from control_analysis.data import load_control_bundle
from control_analysis.jobs import NAMED_PLOT_JOBS, run_doe_group_analysis, run_eval_window_graphs, run_named_plots, run_stats_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run extracted control notebook analysis jobs.")
    parser.add_argument(
        "job",
        nargs="?",
        default="all",
        choices=["all", "eval-windows", "doe-analysis", "named-plots", "stats-report"],
        help="Which extracted notebook job to run.",
    )
    parser.add_argument("--data-dir", default=None, help="Override the data directory used by storage.merge_and_load().")
    parser.add_argument("--output-dir", default="graphs", help="Output directory for eval-window and named plots.")
    parser.add_argument("--analysis-dir", default="graphs/avgs", help="Output directory for DOE group analysis artifacts.")
    parser.add_argument("--max-workers", type=int, default=None, help="Process count for parallel eval-window rendering.")
    parser.add_argument("--plot-names", nargs="*", choices=sorted(NAMED_PLOT_JOBS.keys()), help="Subset of named plots to run.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    bundle = load_control_bundle(data_dir=args.data_dir)
    payload: dict[str, object] = {}

    if args.job in {"all", "eval-windows"}:
        eval_paths = run_eval_window_graphs(data_dir=args.data_dir, output_dir=args.output_dir, max_workers=args.max_workers)
        payload["eval_windows"] = [str(path) for path in eval_paths]

    if args.job in {"all", "doe-analysis"}:
        analysis = run_doe_group_analysis(bundle.df_og, output_dir=args.analysis_dir)
        payload["doe_analysis"] = {
            key: str(value) if hasattr(value, "as_posix") else value.to_dict(orient="records")
            for key, value in analysis.items()
        }

    if args.job in {"all", "named-plots"}:
        payload["named_plots"] = run_named_plots(bundle=bundle, names=args.plot_names, output_dir=args.output_dir)

    if args.job in {"all", "stats-report"}:
        payload["stats_report"] = run_stats_report(bundle=bundle, output_dir=args.analysis_dir)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()