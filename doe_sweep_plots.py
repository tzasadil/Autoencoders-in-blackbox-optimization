import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import storage


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SWEEP_DATA_DIR = os.path.join(ROOT_DIR, "data", "doe_sweep")
SUMMARY_CSV_PATH = os.path.join(SWEEP_DATA_DIR, "summary.csv")
BEST_DOE_CONFIG_PATH = os.path.join(SWEEP_DATA_DIR, "best_doe_config.json")
THESIS_IMG_DIR = os.path.normpath(
    os.path.join(ROOT_DIR, "..", "latexdiff", "new", "en", "img")
)
RANK_HEATMAP_PATH = os.path.join(THESIS_IMG_DIR, "doe_sweep_rank_heatmap.png")
VALUE_HEATMAP_PATH = os.path.join(THESIS_IMG_DIR, "doe_sweep_value_heatmap.png")
BAR_CHART_PATH = os.path.join(THESIS_IMG_DIR, "doe_sweep_bar.png")


def final_value(values):
    array = np.asarray(values)
    if array.size == 0:
        return np.inf
    return float(array[-1])


def parse_model(model_name):
    prefix = "doe_"
    n_samples, latent_dim = model_name.removeprefix(prefix).split("_")
    return int(n_samples), int(latent_dim)


def load_ranked_results():
    df = storage.merge_and_load(data_dir=SWEEP_DATA_DIR)
    if df is None or df.empty:
        raise RuntimeError("No DOE sweep data found in data/doe_sweep")
    ranked = df.copy()
    ranked["final_value"] = ranked["vals"].apply(final_value)
    ranked[["n_samples", "latent_dim"]] = ranked["model"].apply(
        lambda model: pd.Series(parse_model(model))
    )
    ranked["problem_rank"] = ranked.groupby(["function", "dim", "instance"])[
        "final_value"
    ].rank(method="average", ascending=True)
    return ranked


def build_summary(ranked):
    summary = (
        ranked.groupby(["model", "n_samples", "latent_dim"], as_index=False)
        .agg(
            average_rank=("problem_rank", "mean"),
            mean_final_value=("final_value", "mean"),
            median_final_value=("final_value", "median"),
            run_count=("final_value", "size"),
        )
        .sort_values(["average_rank", "mean_final_value", "model"])
        .reset_index(drop=True)
    )
    return summary


def save_best_config(best_row):
    best_config = {
        "model": best_row["model"],
        "label": "best_doe",
        "n_samples": int(best_row["n_samples"]),
        "latent_dim": int(best_row["latent_dim"]),
        "selection_metric": "average_rank",
        "average_rank": float(best_row["average_rank"]),
        "mean_final_value": float(best_row["mean_final_value"]),
    }
    with open(BEST_DOE_CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(best_config, handle, indent=2)
    return best_config


def save_heatmap(summary, metric, output_path, title, cbar_label, fmt):
    pivot = summary.pivot(index="n_samples", columns="latent_dim", values=metric)
    plt.figure(figsize=(6.4, 4.8))
    ax = sns.heatmap(pivot, annot=True, fmt=fmt, cmap="viridis_r")
    ax.set_xlabel("latent dimension")
    ax.set_ylabel("number of sampled functions")
    ax.set_title(title)
    ax.collections[0].colorbar.set_label(cbar_label)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_bar_chart(summary, output_path):
    labels = [f"{int(row.n_samples)}/{int(row.latent_dim)}" for row in summary.itertuples()]
    plt.figure(figsize=(8.6, 4.8))
    plt.bar(labels, summary["average_rank"], color="#4c72b0")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("average rank")
    plt.xlabel("samples / latent dimension")
    plt.title("DOE sweep ranking of all tested settings")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main():
    os.makedirs(SWEEP_DATA_DIR, exist_ok=True)
    os.makedirs(THESIS_IMG_DIR, exist_ok=True)

    ranked = load_ranked_results()
    summary = build_summary(ranked)
    summary.to_csv(SUMMARY_CSV_PATH, index=False)
    best_config = save_best_config(summary.iloc[0])

    sns.set_theme(style="whitegrid")
    save_heatmap(
        summary,
        metric="average_rank",
        output_path=RANK_HEATMAP_PATH,
        title="DOE sweep average rank",
        cbar_label="average rank",
        fmt=".2f",
    )
    save_heatmap(
        summary,
        metric="mean_final_value",
        output_path=VALUE_HEATMAP_PATH,
        title="DOE sweep mean final value",
        cbar_label="mean final value",
        fmt=".2e",
    )
    save_bar_chart(summary, BAR_CHART_PATH)

    print(summary.to_string(index=False))
    print(json.dumps(best_config, indent=2))
    print(f"Saved graphs to {THESIS_IMG_DIR}")


if __name__ == "__main__":
    main()