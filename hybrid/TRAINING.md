# HRM × LM Training Guide

This document explains how to launch the hybrid HRM–LM trainer, how datasets are formatted, and which command-line parameters are available.

## Environment Setup

1. Install dependencies with [uv](https://github.com/astral-sh/uv) (preferred):
   ```bash
   uv pip install -r requirements.txt
   ```
2. Ensure project sources are on the Python path when running scripts (examples below set `PYTHONPATH=src`).

> If you prefer a virtual environment, create one manually (e.g. `python -m venv .venv`) and activate it before installing requirements.

## Running the Trainer

### Dry Run (sanity check)
```bash
PYTHONPATH=src uv run python -m hrm_lm.training.train --dry_run 1
```
This command exercises a single forward/backward pass with random data and prints the resulting loss.

### Synthetic Arithmetic Training
```bash
PYTHONPATH=src uv run python -m hrm_lm.training.train \
  --dry_run 0 \
  --dataset synthetic \
  --steps 200 \
  --val_every 50 \
  --save_dir runs/checkpoints \
  --mixed_precision bf16 \
  --grad_clip 1.0
```
Key points:
- `--dataset synthetic` streams batches from the bundled addition dataset.
- Validation and checkpointing are triggered every `val_every` steps when `--save_dir` is supplied.
- Mixed precision (`bf16` or `fp16`) automatically enables `torch.autocast`; gradient clipping applies post-backward.

### Loading a Custom Config
```bash
PYTHONPATH=src uv run python -m hrm_lm.training.train --config path/to/config.yaml --dataset synthetic --dry_run 0
```
The config must follow the structure in `src/hrm_lm/configs/default.yaml`.

## Dataset Format Expectations

The trainer expects an iterator that yields tuples from `pad_batch` with the following tensors:
- `encoder_ids`: shape `(batch, enc_len)` — token ids for the encoder.
- `decoder_input_ids`: shape `(batch, dec_len)` — decoder inputs (usually BOS + target[:-1]).
- `labels`: shape `(batch, dec_len)` — decoder targets with `-100` marking positions to ignore.
- `encoder_mask`: shape `(batch, enc_len)` — 1 where tokens are valid, 0 for padding.
- `decoder_mask`: shape `(batch, dec_len)` — 1 where tokens are valid, 0 for padding.

To add a custom dataset:
1. Provide a tokenizer exposing `pad_id`, `bos_id`, and `eos_id` (see `SimpleTokenizer`).
2. Produce per-example triples `(encoder_ids, decoder_input_ids, labels)` before padding.
3. Reuse `pad_batch` or replicate its padding logic so masks align with padding tokens.
4. Extend `hrm_lm.training.train` to load your dataset name (mirroring the `synthetic` branch).

## Trainer CLI Parameters

| Flag | Default | Description |
| --- | --- | --- |
| `--config` | `src/hrm_lm/configs/default.yaml` | YAML config path describing model/optim/train settings. |
| `--dry_run` | `1` | When `1`, runs a single synthetic batch and exits. Set to `0` for real training. |
| `--dataset` | `None` | Name of dataset loader (currently `synthetic`). Required when `--dry_run 0`. |
| `--steps` | `200` | Number of optimization steps to run during training mode. |
| `--val_every` | `0` | Frequency (in steps) for validation + checkpointing. Disabled when `0`. |
| `--save_dir` | `None` | Directory for checkpoints; created automatically when provided. |
| `--mixed_precision` | `none` | Precision mode: `none`, `bf16`, or `fp16`. Enables autocast when set. |
| `--grad_clip` | `0.0` | L2 gradient clipping norm; disabled when ≤0. |

Additional behavior:
- Validation batches reuse the same sampler as training (`val_iterator`).
- Checkpoints persist model weights and serialized config (`cfg`).
- Mixed precision uses `torch.cuda.amp.GradScaler` for `fp16` on CUDA devices.

## Useful Shortcuts

- Regenerate synthetic dataset vocabulary: `build_synthetic_dataset()` in `hrm_lm.data.synthetic`.
- Run tests (after installing dependencies):
  ```bash
  PYTHONPATH=src uv run pytest -q
  ```
- Generate samples with the current checkpoint:
  ```bash
  PYTHONPATH=src uv run python -m hrm_lm.inference.generate --prompt "What is 12 + 7?" --dataset synthetic
  ```
