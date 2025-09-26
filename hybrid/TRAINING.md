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

### Encoder backends

The hybrid HRM stack can run with three language backbones:

- `mamba2` (default) – fast state-space encoder with kernel-size padding guard.
- `rwkv6` – linear-time RWKV-style recurrent encoder tuned for stability on very long contexts.
- `transformer` – standard attention stack (quadratic in sequence length).

Set the desired backend in your config (`model.encoder.backend`) or override it per run using `--encoder_backend`.

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

Large raw text dumps (for example Wikipedia) can be chunked locally before feeding the standard converter:

```bash
# Slice long articles into 1k-token windows with 64-token overlap using all CPU cores
python scripts/chunk_text_dataset.py \
  --input datasets/debasishraychawdhuri-wikipedia_clean_5GB/clean_wikipedia_for_autocorrect.txt \
  --output datasets/wiki_chunks/wiki_1024.jsonl \
  --tokenizer tokenizer.json \
  --target-length 1024 \
  --stride 64 \
  --batch-size 256 \
  --num-proc 30

# Convert the chunked JSONL into encoder/decoder triples for training
python scripts/prepare_language_dataset.py \
  --source datasets/wiki_chunks \
  --dest datasets/wiki_chunks/processed \
  --tokenizer datasets/wiki_chunks/processed/tokenizer.json \
  --tokenizer-num-threads 16 \
  --tokenizer-batch-size 256 \
  --max-seq-len 1024 \
  --val-ratio 0.02
```

`chunk_text_dataset.py` accepts both hub identifiers and local tokenizer files. Point `--tokenizer` at an existing `tokenizer.json` to keep vocabulary consistent while trimming articles down to GPU-friendly sequence lengths.

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
Pass `--reset_progress` to load only the weights from that checkpoint while restarting optimizer state, warmup, and step counters from zero.

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
| `--encoder_backend` | config value | Override the encoder stack (`transformer`, `mamba2`, or `rwkv6`). |
| `--batch_size` | config value | Overrides batch size used for training/validation batches. |
| `--optimizer` | `adamw` | Optimizer choice (`adamw` or `adamw_8bit`; the latter requires `bitsandbytes`). |
| `--learning_rate` | config value | Override optimizer learning rate. |
| `--lr_scheduler` | `cosine` | Post-warmup decay strategy: `cosine`, `linear`, or `constant`. |
| `--eval_batch_size` | matches train batch | Override validation batch size (falls back to training batch when omitted). |
| `--warmup_steps` | `0` | Linear LR warmup steps before applying the selected scheduler. |
| `--steps` | config value | Total optimization steps; overrides `--epochs` when >0. |
| `--epochs` | `0` | Number of epochs (computed from dataset size) when `--steps` ≤ 0. |
| `--val_every` | `0` | Validation/checkpoint frequency in steps (disabled when `0`). |
| `--eval_loss_patience` | `3` | Stop training after this many consecutive validation loss increases (`0` disables early stop). |
| `--patience_grace_steps` | `0` | Minimum global step before patience counting starts (useful to ignore noisy post-warmup evals). |
| `--save_dir` | `None` | Legacy manual checkpoint directory (overridden by `--run_name`). |
| `--run_name` | `None` | Creates `runs/<run-name>/checkpoints/` (required for `--save_best_model`). |
| `--checkpoint_limit` | `0` | Maximum number of `step_*.pt` checkpoints to retain (FIFO). `0` disables rotation. |
| `--save_best_model` | *flag* | When set, saves lowest-validation-loss model to `runs/<run-name>/best-model/best.pt`. |
| `--max_seq_len` | config value | Truncate encoder/decoder sequences to this length. |
| `--log_steps` | `10` | Emit Rich-formatted training metrics every N steps. |
| `--dataset_workers` | `0` | Number of worker processes for JSONL loading (`0`/`1` = single-process). |
| `--max_val_samples` | `0` | Cap the number of validation samples per eval sweep (`0` uses the full split). |
| `--hrm_gate_warmup_steps` | `0` | Hold the HRM bridge gate at zero for the first N steps before blending HRM latents into the decoder. |
| `--lr_min_ratio` | `0.0` | Floor multiplier for cosine/linear schedulers (e.g. `0.05` keeps LR ≥5% of the base). |
| `--mixed_precision` | `none` | Precision mode: `none`, `bf16`, or `fp16` (fp16 requires CUDA). |
| `--grad_clip` | `0.0` | L2 gradient clipping norm (disabled when ≤0). |
| `--reset_progress` | *flag* | Load checkpoint weights but restart from step 0 (ignores optimizer/scaler state). |

Additional notes:
- Checkpoint payloads include optimizer state, GradScaler state (when fp16), and the best validation loss so auto-resume reproduces optimizer momentum.
- `final.pt` always reflects the last completed step, regardless of the checkpoint limit.
- Dataset loading now reports progress heartbeats (every ~200k samples) and can be parallelized with `--dataset_workers` for faster ingest on large corpora.
- Extremely large corpora automatically stream from disk using offset indexes to avoid exhausting RAM (worker processes apply only when caching).
- Validation runs over the entire `val.jsonl`, averaging loss across all batches for accurate metrics (override with `--max_val_samples`).
- Validation loops now reuse the active mixed-precision autocast context, so bf16/fp16 runs no longer inflate memory during evaluation sweeps.
- `--eval_loss_patience` provides an automatic safety stop when validation loss rises repeatedly; lower values trigger earlier restarts and `0` keeps training regardless of the trend.
- Combine `--patience_grace_steps` with the early-stop flag to ignore the first few post-warmup evaluations until metrics stabilize.
- `--hrm_gate_warmup_steps` lets you pretrain the language pathway before blending HRM latents; once the warmup window passes the gate opens automatically.
- Use `--lr_min_ratio` to keep cosine/linear schedules from collapsing the learning rate during very long runs.
- When a CUDA kernel times out the trainer now catches the failure, flushes caches safely, and retries the step after a short pause; the warning message tells you the retry is happening.

### Gated HRM Language Warmup (Stage A)
Use this recipe to reproduce the initial English-only stabilization run before introducing reasoning curricula:

1. Make sure `datasets/wiki_chunks/processed` (≈1.22B tokens) exists via `scripts/prepare_language_dataset.py` as described above.
2. Launch the gated run with AdamW 8-bit, HRM gate warmup, and patience grace period:

```bash
python -m hrm_lm.training.train \
  --dataset datasets/wiki_chunks/processed/ \
  --batch_size 22 \
  --steps 308212 \
  --learning_rate 0.00025 \
  --warmup_steps 4000 \
  --lr_scheduler cosine \
  --lr_min_ratio 0.05 \
  --grad_clip 5.0 \
  --val_every 1500 \
  --run_name hrm-gated-pretrain \
  --checkpoint_limit 5 \
  --mixed_precision bf16 \
  --eval_batch_size 22 \
  --log_steps 1 \
  --dataset_workers 30 \
  --save_best_model \
  --optimizer adamw_8bit \
  --max_seq_len 512 \
  --max_val_samples 20000 \
  --eval_loss_patience 3 \
  --patience_grace_steps 10000 \
  --hrm_gate_warmup_steps 8000
```

3. Allow the run to continue until validation loss plateaus (≥30k steps recommended) before enabling HRM auxiliaries or mixing reasoning datasets.

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
