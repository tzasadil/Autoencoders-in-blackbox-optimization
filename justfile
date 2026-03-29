
run:
  uv run python main.py

sweep-vae:
  uv run python vae_sweep.py

optim:
  uv run optim

coco:
  uv run coco

compare-with-best-vae:
  just sweep-vae
  just optim
