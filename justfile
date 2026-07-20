run: sweep-doe plots-doe-sweep optim coco-external coco graphs

run-main:
    uv run python main.py

graphs:
    uv run control-graphs all

sweep-doe:
    uv run python doe_sweep.py

plots-doe-sweep:
    uv run python doe_sweep_plots.py

optim:
    uv run optim

# Build COCO post-processing for this repo's own experiment outputs.
coco:
    uv run coco

# Build the external reference bundle under ppdata/external_refs.
coco-external:
    uv run python build_external_coco_bundle.py
