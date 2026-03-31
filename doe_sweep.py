import json
import os

import numpy as np
import pandas as pd

import main
import storage


DOE_SAMPLE_VARIANTS = [1, 2, 4, 8, 16]
DOE_LATENT_VARIANTS = [1 ,2, 4, 8, 16]
SWEEP_DATA_DIR = os.path.join("data", "doe_sweep")
SWEEP_NOTE = "doe_sweep"
SWEEP_RESULT_PREFIX = "doe_sweep__"
SWEEP_PROBLEM_INFO = "function_indices:1-24 dimensions:5 instance_indices:1"
BEST_DOE_CONFIG_PATH = os.path.join(SWEEP_DATA_DIR, "best_doe_config.json")
SUMMARY_CSV_PATH = os.path.join(SWEEP_DATA_DIR, "summary.csv")


def build_sweep_configs():
    configs = []
    for n_samples in DOE_SAMPLE_VARIANTS:
        for latent_dim in DOE_LATENT_VARIANTS:
            configs.append([None, 4, None, main.build_doe_model(n_samples, latent_dim)])
    return configs


def final_value(values):
    array = np.asarray(values)
    if array.size == 0:
        return np.inf
    return float(array[-1])


def summarize_results(df):
    ranked = df.copy()
    ranked["final_value"] = ranked["vals"].apply(final_value)
    ranked["problem_rank"] = ranked.groupby(["function", "dim", "instance"])["final_value"].rank(
        method="average", ascending=True
    )
    summary = (
        ranked.groupby("model", as_index=False)
        .agg(
            average_rank=("problem_rank", "mean"),
            # mean_final_value=("final_value", "mean"),
            # median_final_value=("final_value", "median"),
            # run_count=("final_value", "size"),
        )
        .sort_values(["average_rank"])#, "mean_final_value", "model"])
        .reset_index(drop=True)
    )
    return summary


def extract_best_config(model_name):
    prefix = "doe_"
    n_samples, latent_dim = model_name.removeprefix(prefix).split("_")
    return {
        "model": model_name,
        "label": "best_doe",
        "n_samples": int(n_samples),
        "latent_dim": int(latent_dim),
        "selection_metric": "average_rank",
    }


def run_sweep():
    os.makedirs(SWEEP_DATA_DIR, exist_ok=True)
    df_existing = storage.merge_and_load(data_dir=SWEEP_DATA_DIR)

    results = main.run(
        df=df_existing,
        configs=build_sweep_configs(),
        problem_info=SWEEP_PROBLEM_INFO,
        data_dir=SWEEP_DATA_DIR,
        result_folder_prefix=SWEEP_RESULT_PREFIX,
        experiment_note=SWEEP_NOTE,
        include_best_doe=False,
    )
    summary = summarize_results(results)
    summary.to_csv(SUMMARY_CSV_PATH, index=False)

    best_config = extract_best_config(summary.iloc[0]["model"])
    with open(BEST_DOE_CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(best_config, handle, indent=2)

    print(summary.head())
    print(f"Saved best DOE config to {BEST_DOE_CONFIG_PATH}")
    return summary, best_config


if __name__ == "__main__":
    run_sweep()
