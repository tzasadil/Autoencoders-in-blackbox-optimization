run-main:
    uv run python main.py

graphs:
    uv run control-graphs all

sweep-doe:
    uv run python doe_sweep.py

run: sweep-doe run-main

plots-doe-sweep:
    uv run python doe_sweep_plots.py

optim:
    uv run optim

coco:
    uv run coco

coco-external:
    uv run python build_external_coco_bundle.py
