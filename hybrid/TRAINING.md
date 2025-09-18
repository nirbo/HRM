# HRM × LM Training Guide

This document explains how to launch the hybrid HRM–LM trainer, how datasets are formatted, and which command-line parameters are available.

## Environment Setup

1. Install dependencies with [uv](https://github.com/astral-sh/uv) (preferred):
   ```bash
   uv pip install -r requirements.txt
   ```
2. Ensure project sources are on the Python path when running scripts (examples below set `PYTHONPATH=src`).

> If you prefer a traditional virtual environment, create one manually (e.g. `python -m venv .venv`) and activate it before installing requirements.

## Running the Trainer

## Dataset Preparation

To convert parquet dumps (like the FineWeb derivative) into HRM-LM-ready token triplets, run:

```bash
PYTHONPATH=src uv run python scripts/prepare_language_dataset.py \
  --source datasets/anothy1-fineweb-edu-cleaned-simplified \
  --dest datasets/anothy1-fineweb-edu-cleaned-simplified/processed \
  --vocab-size 128000 \
  --max-seq-len 256 \
  --val-ratio 0.02
```

If `--tokenizer` points to an existing `tokenizer.json`, it is reused; otherwise a new BPE tokenizer (Hugging Face format) is trained and saved alongside the processed dataset. The script writes `train.jsonl`, `val.jsonl`, the tokenizer JSON, and `meta.json` containing padding IDs and vocab size. Each sample stores `encoder_ids`, `decoder_input_ids`, and `labels` arrays of tokens ready for loading.
Optional flags: `--tokenizer-num-threads` to cap CPU threads, `--tokenizer-batch-size` to control encoding batch size, and `--max-files` for quick smoke tests.

Large datasets can be processed in parallel batches and merged without doubling disk usage via:

```bash
PYTHONPATH=src python scripts/merge_prepared_batches.py \
  --batches datasets/redpj/batches \
  --output-dir datasets/redpj/combined
```

This command streams each chunk’s `train.jsonl`/`val.jsonl` into consolidated files, writes an aggregated `meta.json`, and deletes chunk directories by default (pass `--keep-chunks` to retain them).

For QA-style sources (SQuAD, TriviaQA, etc.), normalize schemas first:
```bash
python scripts/normalize_qa_dataset.py \
  --input squad/train.jsonl \
  --output datasets/qa/squad_prompts.jsonl \
  --question-field data.question \
  --answer-field data.answers.text[0]
```
The tool auto-detects `.json`, `.jsonl`, `.parquet`, or `.arrow` inputs and emits prompt/response JSONL using configurable templates. By default prompts are just questions and responses just answers; include flags such as `--context-field data.context --prompt-template "{question}\nContext: {context}"` when you need extra context.


### Dry Run (sanity check)
```bash
PYTHONPATH=src uv run python -m hrm_lm.training.train --dry_run
```
Performs a single forward/backward pass with random data and prints the loss.

### Synthetic Arithmetic Training with Checkpoint Management
```bash
PYTHONPATH=src uv run python -m hrm_lm.training.train \
  --dataset synthetic \
  --batch_size 4 \
  --learning_rate 2.5e-4 \
  --warmup_steps 50 \
  --steps 200 \
  --val_every 50 \
  --run_name arithmetic-demo \
  --checkpoint_limit 3 \
  --save_best_model \
  --mixed_precision bf16 \
  --log_steps 20 \
  --grad_clip 1.0
```
Key behaviors:
- `--run_name` creates `runs/arithmetic-demo/checkpoints/` and stores artifacts there.
- The trainer automatically resumes from the most recent checkpoint inside that directory (no extra flags needed).
- `--checkpoint_limit 3` keeps only the 3 newest `step_*.pt` files (FIFO rotation) while leaving `final.pt` intact.
- `--save_best_model` maintains `runs/arithmetic-demo/best-model/best.pt`, updated whenever validation loss improves. Matching `best.yaml`, `tokenizer.json`, and `meta.json` files live beside the weights for reproducibility.
- Each `step_*.pt` and `final.pt` checkpoint now emits a sibling `.yaml` config plus copies of `tokenizer.json` and `meta.json` when available.
- `--optimizer` defaults to `adamw`; pass `--optimizer adamw_8bit` (requires `pip install bitsandbytes`) for 8-bit weights.
- `--learning_rate`, `--warmup_steps`, and `--lr_scheduler` control LR warmup and decay (defaults to cosine).
- `--eval_batch_size` lets you shrink or expand validation throughput independently of training batches (defaults to the training batch size).
- `--log_steps` governs how often training metrics are emitted with Rich-formatted output.
- `--dataset <dir>` can point to a processed corpus (with `train/val.jsonl` and `meta.json`); metadata adjusts vocab size automatically.
- Mixed precision (`bf16` or `fp16`) engages `torch.autocast`; gradient clipping applies after each backward pass.

### Loading a Custom Config
```bash
PYTHONPATH=src uv run python -m hrm_lm.training.train --config path/to/config.yaml --dataset synthetic --run_name custom-run
```
The config must follow the structure found in `src/hrm_lm/configs/default.yaml`.

## Auto-Resume Behavior

Whenever a checkpoint directory already contains saved models, the trainer loads the latest checkpoint (by step number) before continuing. If the stored step count is greater than or equal to the requested `--steps`, the run exits immediately to avoid duplicate work.

## Dataset Format Expectations

Batches must follow the structure produced by `pad_batch`:
- `encoder_ids`: `(batch, enc_len)` token IDs for the encoder.
- `decoder_input_ids`: `(batch, dec_len)` decoder inputs (usually BOS + answer without EOS).
- `labels`: `(batch, dec_len)` decoder targets (answer without BOS, with EOS).
- `encoder_mask`: `(batch, enc_len)` mask with 1 for valid tokens, 0 for padding.
- `decoder_mask`: `(batch, dec_len)` mask with 1 for valid tokens, 0 for padding.

To add a custom dataset:
1. Provide a tokenizer exposing `pad_id`, `bos_id`, and `eos_id` (see `SimpleTokenizer`).
2. Produce per-example triples `(encoder_ids, decoder_input_ids, labels)` prior to padding.
3. Reuse `pad_batch` or duplicate its padding/masking logic so inputs align with masks.
4. Extend `hrm_lm.training.train` with a new dataset branch similar to `synthetic`.

## Trainer CLI Parameters

| Flag | Default | Description |
| --- | --- | --- |
| `--config` | `src/hrm_lm/configs/default.yaml` | YAML config describing model/optim/train settings. |
| `--dry_run` | `1` | When `1`, runs a single synthetic batch and exits; set to `0` for training. |
| `--dataset` | `None` | Dataset loader name (currently `synthetic`). Required when training. |
| `--batch_size` | config value | Overrides batch size used for training/validation batches. |
| `--optimizer` | `adamw` | Optimizer choice (`adamw` or `adamw_8bit`; the latter requires `bitsandbytes`). |
| `--learning_rate` | config value | Override optimizer learning rate. |
| `--lr_scheduler` | `cosine` | Post-warmup decay strategy: `cosine`, `linear`, or `constant`. |
| `--eval_batch_size` | matches train batch | Override validation batch size (falls back to training batch when omitted). |
| `--warmup_steps` | `0` | Linear LR warmup steps before applying the selected scheduler. |
| `--steps` | config value | Total optimization steps; overrides `--epochs` when >0. |
| `--epochs` | `0` | Number of epochs (computed from dataset size) when `--steps` ≤ 0. |
| `--val_every` | `0` | Validation/checkpoint frequency in steps (disabled when `0`). |
| `--save_dir` | `None` | Legacy manual checkpoint directory (overridden by `--run_name`). |
| `--run_name` | `None` | Creates `runs/<run-name>/checkpoints/` (required for `--save_best_model`). |
| `--checkpoint_limit` | `0` | Maximum number of `step_*.pt` checkpoints to retain (FIFO). `0` disables rotation. |
| `--save_best_model` | *flag* | When set, saves lowest-validation-loss model to `runs/<run-name>/best-model/best.pt`. |
| `--max_seq_len` | config value | Truncate encoder/decoder sequences to this length. |
| `--log_steps` | `10` | Emit Rich-formatted training metrics every N steps. |
| `--dataset_workers` | `0` | Number of worker processes for JSONL loading (`0`/`1` = single-process). |
| `--mixed_precision` | `none` | Precision mode: `none`, `bf16`, or `fp16` (fp16 requires CUDA). |
| `--grad_clip` | `0.0` | L2 gradient clipping norm (disabled when ≤0). |

Additional notes:
- Checkpoint payloads include optimizer state, GradScaler state (when fp16), and the best validation loss so auto-resume reproduces optimizer momentum.
- `final.pt` always reflects the last completed step, regardless of the checkpoint limit.
- Dataset loading now reports progress heartbeats (every ~200k samples) and can be parallelized with `--dataset_workers` for faster ingest on large corpora.

## Useful Shortcuts

- Regenerate synthetic dataset vocabulary: `build_synthetic_dataset()` in `hrm_lm.data.synthetic`.
- Run the unit test suite:
  ```bash
  PYTHONPATH=src uv run pytest -q
  ```
- Generate samples with the current checkpoint:
  ```bash
  PYTHONPATH=src uv run python -m hrm_lm.inference.generate --prompt "What is 12 + 7?" --dataset synthetic --run_name arithmetic-demo
  ```
