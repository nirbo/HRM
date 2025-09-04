# Repository Guidelines

## Project Structure & Module Organization
- `models/`: HRM core (`hrm/`), transformer frontend, adapters, losses, layers, and common utilities.
- `dataset/`: dataset builders for ARC, Sudoku, and Maze; writes under `data/`.
- `config/`: Hydra/OmegaConf YAMLs (experiment and architecture settings).
- `utils/`: shared helpers (e.g., `utils/functions.py`).
- Top-level scripts: `pretrain.py`, `evaluate.py`, `train_hybrid.py`, `evaluate_hybrid.py`, `puzzle_dataset.py`.
- Assets and tools: `assets/` (images/js), `puzzle_visualizer.html`, `arc_eval.ipynb`.

## Build, Test, and Development Commands
- Create env and install: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Login to W&B: `wandb login`
- Build datasets (example): `python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000`
- Single-GPU quick run: `OMP_NUM_THREADS=8 python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=2000 eval_interval=200 lr=7e-5`
- Multi-GPU: `OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=...`
- Evaluate: `OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 evaluate.py checkpoint=<PATH>`
Note: CUDA and FlashAttention setup is in `README.md`.

## Coding Style & Naming Conventions
- Python (PEP 8), 4-space indent, limit lines to ~100 chars.
- Names: `snake_case` for files/functions, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- Type hints where practical; docstrings for public functions/classes.
- Config via Hydra: prefer `key=value` CLI overrides over hardcoding (e.g., `pretrain.py data_path=... lr=1e-4`).

## Testing Guidelines
- No formal unit test suite yet. Use smoke tests:
  - Dataset build completes and writes to `data/...`.
  - Short training run: `epochs=100` and `eval_interval=50` on a small dataset to verify loss decreases.
  - Model evaluation runs and logs `eval/exact_accuracy` to W&B.
- For ARC evaluation details, use `arc_eval.ipynb` after `evaluate.py`.

## Commit & Pull Request Guidelines
- Commits: concise, imperative mood ("Add", "Fix"), scope prefix when useful (e.g., `dataset:`, `models:`, `docs:`), reference issues with `#123`.
- PRs must include: purpose/summary, minimal repro commands, config diffs (Hydra overrides), and screenshots or W&B run links for metrics. Update `README.md` and `config/` when behavior or defaults change.
- Do not commit datasets or checkpoints; keep secrets (HF/W&B tokens) out of git.

## Security & Configuration Tips
- Store credentials in env vars (e.g., `WANDB_API_KEY`) and avoid committing `.env` files.
- Large artifacts live outside the repo (e.g., `data/`, `checkpoints/`) and should be gitignored locally.
