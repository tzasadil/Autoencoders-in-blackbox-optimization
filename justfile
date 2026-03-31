
run:
  uv run python main.py

sweep-doe:
  uv run python doe_sweep.py

plots-doe-sweep:
  uv run python doe_sweep_plots.py

optim:
  uv run optim

coco:
  uv run coco


