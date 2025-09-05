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

## Hybrid Integration Status (Work Log)
- Branch: working on `dev` (tracking `origin/dev`). `main` remains the upstream base.
- New files: `models/transformer_frontend.py`, `models/adapters.py`, `models/hybrid_hrm_transformer.py`, `evaluate_hybrid.py`, `train_hybrid.py`, `utils/reasoning.py`, `utils/text_codec.py`, `scripts/test_hybrid_routing.py`, `dataset/build_text_dataset.py`.
- README: added Hybrid section and [REASON] JSON examples. `requirements.txt`: added `transformers`.

### What Works Now
- Transformer front-end: Hugging Face causal LM wrapper with `[REASON]`/`[ENDREASON]` special tokens.
- Hybrid wrapper: auto‑detects adapter dims; instantiates HRM on transformer’s device; exposes `load_hrm_checkpoint(path)`.
- Routing: extracts `[REASON]... [ENDREASON]`; parses JSON; supports tasks:
  - `calc` (stub expression evaluator for smoke tests)
  - `text` (byte‑level prompt routed through HRM; requires HRM vocab_size ≥ 257)
  - `sudoku` (example domain; optional)
- Tests: `python scripts/test_hybrid_routing.py` validates adapter dims + routing without HF/flash‑attn.

### How To Run Quickly
- End‑to‑end demo: `python evaluate_hybrid.py` (shows calc + example domain). For coding/math, use:
  - `demo = 'Answer: [REASON] {"task":"text","prompt":"2+2*(3-1) = "} [ENDREASON]'`
  - Load HRM first: `model.load_hrm_checkpoint('checkpoints/<project>/<run>/step_<N>')`

### Build Byte‑Level Text Dataset (for coding/math/science)
- Prepare JSONL lines: `{"prompt":"<input>", "target":"<desired output>"}` under `data/raw/text_tasks/{train,test}.jsonl`.
- Build: `python dataset/build_text_dataset.py --input-dir data/raw/text_tasks --output-dir data/text-512 --seq-len 512`
- Metadata uses `vocab_size=257`, `seq_len=512`. Tokens: PAD=0, bytes=1..256.

### Train HRM (non‑autoregressive + ACT)
- Single node (example):
  - `OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/text-512`
- Ensure Hydra overrides match dataset: `arch.*`, `seq_len`, `vocab_size=257`, batch sizes, LR, and `halt_max_steps`.
- Check W&B metrics; checkpoints saved under `checkpoints/<project>/<run>/step_<N>`.

### Known Issues / Notes
- Transformers may warn about ignored flags (safe to ignore). Large models will download on first run.
- Ensure FlashAttention and CUDA/PyTorch versions match. Device mismatches are fixed by constructing HRM on the transformer’s device.

### Next Actions (Tomorrow)
- Train a small byte‑level HRM on a toy text set; verify `{"task":"text"}` path end‑to‑end.
- Add streaming routing via HF StoppingCriteria (intercept [REASON] during generation).
- Optional: expand handler registry (math scratchpad, code I/O schema), and add logging/metrics hooks.

## Reasoning Engine Roadmap (Coding/Math/Science)

- Current behavior: the transformer handles natural language, extracts `[REASON]... [ENDREASON]`, HRM computes the result, and the transformer verbalizes it. Adapters exist but are not yet used to exchange latent states.

- Build a byte‑level text dataset:
  - Prepare JSONL under `data/raw/text_tasks/{train,test}.jsonl` with `{"prompt":"...","target":"..."}`.
  - Build processed data: `python dataset/build_text_dataset.py --input-dir data/raw/text_tasks --output-dir data/text-512 --seq-len 512`.
  - Dataset semantics: `vocab_size=257` (PAD=0, bytes=1..256), `seq_len` typically 512–1024.

- Train HRM on text (non‑autoregressive + ACT):
  - Single node: `OMP_NUM_THREADS=8 python pretrain.py data_path=data/text-512 epochs=2000 eval_interval=200 vocab_size=257`.
  - Multi‑GPU: `OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/text-512 epochs=2000 eval_interval=200`.
  - Ensure Hydra overrides match the dataset: `vocab_size=257`, `seq_len=512`, `halt_max_steps` (e.g., 8–16), and scale `hidden_size/num_heads/batch_size/lr` to budget.
  - Monitor loss and eval metrics; checkpoints at `checkpoints/<project>/<run>/step_<N>`.

- Evaluate `task:"text"` end‑to‑end:
  - `python evaluate_hybrid.py --device cuda:0 --hrm-checkpoint checkpoints/<project>/<run>/step_<N> --prompt 'Answer: [REASON] {"task":"text","prompt":"2+2*(3-1) = "} [ENDREASON]'`.
  - Expect meaningful outputs only after HRM text training.

- Quality of life for outputs:
  - Add deterministic generation: small `max_new_tokens`, `do_sample=False`, `temperature=0.0` to reduce chatter.
  - Optionally trim echoed prompt and show only the continuation.
  - Use an EOS convention in targets and stop decoding at EOS.

- Deeper hybrid coupling (next iterations):
  - Use `EncoderAdapter` to project transformer hidden states of the `[REASON]` span into HRM to seed computation; use `DecoderAdapter` to project HRM states back to guide the transformer.
  - Implement streaming routing via HF `StoppingCriteria` to intercept `[REASON]` during generation, invoke HRM, then resume decoding.
  - Expand handlers: keep `{"task":"text"}` as default; optionally add specialized math/code schemas with structured I/O.

- Recommended config presets:
  - Data: `vocab_size=257`, `seq_len` 512–1024 depending on tasks.
  - HRM: `halt_max_steps` 8–16, `H/L_layers` 4–8, `hidden_size` 512–1024, `num_heads` 8–16.
  - Training: watch `train/loss` and task‑specific eval metrics (exact match/BLEU, etc.).
