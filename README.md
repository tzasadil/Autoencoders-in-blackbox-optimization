# VAEs in black-box optimization

Code for the thesis experiments: DoE2Vec-style VAEs as CMA-ES preselection surrogates on [COCO/BBOB](https://github.com/numbbo/coco), compared against GP, ELM, nearest-neighbor, and a few oracle/selector controls.

Results land in `data/` (CSV). COCO post-processing goes to `ppdata/`, plots to `graphs/`.

Needs Python 3.11, [uv](https://github.com/astral-sh/uv), and [just](https://github.com/casey/just).

```bash
uv sync
just run
```

`just run` does the full chain: DoE sweep, sweep plots, optimization, external COCO refs, COCO post-processing, analysis graphs.

Individual targets:

- `just sweep-doe` / `just plots-doe-sweep`: VAE hyperparameter sweep and its plots
- `just optim` / `just run-main`: full optimization (`optim` entrypoint or `main.py`)
- `just coco-external` / `just coco`: external reference bundle / post-process this repo's runs
- `just graphs`: control / thesis analysis plots

Default BBOB slice is functions 1-24, dims 2/5/10, instances 1-10, budget `250 * dim`. `just optim` runs one worker per model by default.

Optional env vars:

- `BBOB_PROBLEM_INFO`: COCO filter string
- `BBOB_MODEL_FILTER`: run only one model
- `BBOB_DATA_DIR`: CSV output directory
- `BBOB_SKIP_PLOT=1` / `BBOB_SKIP_COCO=1`: skip ranking plots or COCO post-processing

Rough layout: `main.py` / `evo.py` / `models.py` for the experiment loop, `doe2vec/` for the VAE surrogate, `control_analysis/` for the graph CLI, plus `data/`, `exdata/`, `ppdata/`, `graphs/`.
