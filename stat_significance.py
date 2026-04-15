import argparse
from pathlib import Path

from control_analysis import DEFAULT_EQUIVALENCE_DELTA, DEFAULT_EQUIVALENCE_SWEEP, load_control_bundle, write_stats_report


DEFAULT_OUTPUT_DIR = Path("graphs/avgs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Friedman/Nemenyi significance table for the thesis.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for exported significance artifacts.")
    parser.add_argument(
        "--equivalence-delta",
        type=float,
        default=DEFAULT_EQUIVALENCE_DELTA,
        help="Paired TOST equivalence margin for DOE vs no surrogate on per-block average ranks.",
    )
    parser.add_argument(
        "--equivalence-sweep",
        type=float,
        nargs="*",
        default=DEFAULT_EQUIVALENCE_SWEEP,
        help="Candidate equivalence margins to sweep when looking for the first margin where TOST equivalence holds.",
    )
    args = parser.parse_args()

    bundle = load_control_bundle()
    report = write_stats_report(
        bundle.df_og,
        output_dir=args.output_dir,
        equivalence_delta=args.equivalence_delta,
        equivalence_sweep=args.equivalence_sweep,
    )

    print(f"Wrote {report['latex_path']}")
    print(f"Wrote {report['json_path']}")
    print(f"Friedman chi-square: {report['friedman_statistic']:.4f}")
    print(f"Friedman p-value: {report['friedman_pvalue']:.6g}")
    print(f"First equivalent margin: {report['first_equivalent_margin']}")


if __name__ == "__main__":
    main()
