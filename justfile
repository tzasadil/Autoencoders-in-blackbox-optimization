
run:
  uv run python main.py

sweep-doe:
  uv run python doe_sweep.py

plots-doe-sweep:
  uv run python doe_sweep_plots.py

sweep-vae:
  just sweep-doe

optim:
  uv run optim

coco:
  uv run coco

compare-with-best-doe:
  just sweep-doe
  just optim

compare-with-best-vae:
  just compare-with-best-doe
