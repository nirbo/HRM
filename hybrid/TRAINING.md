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

The trainer now wires optimizers, schedulers, and losses through [kozistr/pytorch_optimizer](https://github.com/kozistr/pytorch_optimizer). The defaults in `src/hrm_lm/configs/*.yaml` use `adamw` + `cosine` + cross-entropy, but every component can be overridden either in the YAML files or directly on the command line. All selections (including kwargs) are stored in checkpoints so resumed runs continue with the same configuration.

### Configuration Fields

```yaml
optim:
  name: adamw
  lr: 7.5e-5
  weight_decay: 0.01
  betas: [0.9, 0.95]
  kwargs: {}
loss:
  name: cross_entropy
  kwargs: {}
train:
  lr_scheduler:
    name: cosine
    kwargs:
      min_lr_ratio: 0.02
```

* `optim.name` selects any optimizer listed in the tables below. The library defaults are used unless you supply values under `optim.kwargs` (e.g. `{use_gc: true, weight_decay: 0.01}`).
* `loss.name` can be `cross_entropy` for PyTorch’s built-in loss or any entry from the loss table. Additional arguments (e.g. `label_smoothing`) belong under `loss.kwargs`. The trainer flattens logits/labels to `[tokens, vocab]` and drops any positions with label `-100` before invoking the selected loss.
* `train.lr_scheduler.name` accepts any scheduler from the scheduler table. Warmup is still controlled by `--warmup_steps`/`train.warmup_steps`; if the scheduler natively supports warmup, set it in `lr_scheduler.kwargs` instead.
* `train.grad_accum_steps` defines gradient accumulation. A value of 1 behaves like the legacy trainer; values >1 accumulate that many micro-batches before each optimizer/scheduler update.

### CLI Overrides

```bash
PYTHONPATH=src python -m hrm_lm.training.train \
  --config src/hrm_lm/configs/moe.yaml \
  --optimizer ranger21 \
  --optimizer_kwargs '{"use_gc": true, "weight_decay": 0.02}' \
  --lr_scheduler warmup_stable_decay \
  --lr_scheduler_kwargs '{"num_stable_steps": 20000, "min_lr_ratio": 0.05}' \
  --grad_accum_steps 4 \
  --loss bcefocalloss \
  --loss_kwargs '{"gamma": 1.5}'
```

* `--optimizer`, `--lr_scheduler`, and `--loss` fall back to the YAML values when omitted.
* `--optimizer_kwargs`, `--lr_scheduler_kwargs`, and `--loss_kwargs` accept JSON or Python dict literals. They are merged with config-provided values, letting you tweak a single hyperparameter without rewriting the entire dictionary.
* `--grad_accum_steps` overrides `train.grad_accum_steps`; when >1 the trainer divides each micro-batch loss by the accumulation factor, accumulates gradients across micro-batches, and only clips/updates/schedules on effective optimizer steps.
* Checkpoints store the resolved names/kwargs and restore the scheduler state on resume. If an older checkpoint lacks scheduler metadata the trainer prints a warning and restarts the schedule from step 0.

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

If `--tokenizer` points to an existing `tokenizer.json`, it is reused; otherwise a new BPE tokenizer (Hugging Face format) is trained and saved alongside the processed dataset. The script now writes `train.jsonl`, `val.jsonl`, `test.jsonl`, the tokenizer JSON, and `meta.json` containing padding IDs, vocab size, and split counts. Each sample stores `encoder_ids`, `decoder_input_ids`, and `labels` arrays of tokens ready for loading.
Optional flags: `--tokenizer-num-threads` to cap CPU threads, `--tokenizer-batch-size` to control encoding batch size, `--max-files` for quick smoke tests, and `--test-ratio` to reserve an explicit test hold-out when the raw source lacks an official test split (defaults to the validation ratio). The converter auto-detects Hugging Face style file names—if the source directory already includes `train`, `validation`, or `test` files (e.g. when pointing directly at `~/.cache/huggingface/datasets/<provider>/<dataset>/<config>/<rev>/`), those splits are preserved verbatim instead of being re-sampled.

### Pulling slices from massive web datasets

For huge corpora such as Nemotron-CC (≈10 TB), clone the repository with `aria2c` support and use `scripts/download_dataset_slice.py` to download manageable slices without duplicating work.

Requirements:
- `aria2c` on PATH (`sudo apt install aria2` or `brew install aria2`).
- Python package `zstandard` installed (already in requirements).

Examples:

```bash
# Grab the first 5 million samples from Nemotron-CC jsonl shards
python scripts/download_dataset_slice.py \
  --index-url https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/data-jsonl.paths.gz \
  --base-url https://data.commoncrawl.org/ \
  --output data/nemotron_cc_slice.jsonl \
  --start 1 \
  --count 5000000 \
  --aria2c-connections 32 \
  --aria2c-split 32

# Pull ~20 GB of high-quality samples into a compressed file
python scripts/download_dataset_slice.py \
  --index-url https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/data-jsonl.paths.gz \
  --base-url https://data.commoncrawl.org/ \
  --pattern "quality=high" \
  --output data/nemotron_high_quality.zst \
  --target-size 20G \
  --compress \
  --compression-level 5
```

Each run writes a metadata JSON alongside the slice (default `<output>.meta.json`) that records start/end sample numbers and source URLs so subsequent runs can skip previously used ranges.

### Cleaning and tokenizing datasets with one command

Use `scripts/prepare_dataset.py` to extract fields from a dataset and optionally convert the result into train/val/test triples ready for the trainer.

**Example – keep only the `text` field**

```bash
python scripts/prepare_dataset.py \
  --source datasets/nemotroncc_high_quality.zst \
  --output-dir datasets/nemotron_clean \
  --fields text
```

This writes `datasets/nemotron_clean/extracted.jsonl` containing records with only the requested field.

**Example – extract and tokenize into triples**

```bash
python scripts/prepare_dataset.py \
  --source datasets/nemotroncc_high_quality.zst \
  --output-dir datasets/nemotron_tokenized \
  --fields text \
  --to-triples \
  --text-field text \
  --tokenizer tokenizer.json \
  --max-seq-len 512 \
  --val-ratio 0.01 \
  --test-ratio 0.01
```

The script produces `extracted.jsonl` plus `train.jsonl`, `val.jsonl`, `test.jsonl`, and `meta.json` containing tokenized triples. If `tokenizer.json` does not exist it is trained automatically (configurable via `--vocab-size`, `--tokenizer-num-threads`, etc.). Subsequent runs reuse those outputs; add `--force-extract` and/or `--force-triples` to redo stages, and `--count-records` if you want progress bars to display total counts and ETA (pre-pass required).

Large datasets can be processed in parallel batches and merged without doubling disk usage via:

```bash
PYTHONPATH=src python scripts/merge_prepared_batches.py \
  --batches datasets/redpj/batches \
  --output-dir datasets/redpj/combined
```

This command streams each chunk’s `train.jsonl` / `val.jsonl` / `test.jsonl` into consolidated files, writes an aggregated `meta.json`, and deletes chunk directories by default (pass `--keep-chunks` to retain them). When you want to blend multiple processed datasets, `merge_prepared_batches.py` also accepts `--sources`, `--weights`, and optional `--train-samples` / `--val-samples` / `--test-samples` quotas so every split ends up with balanced sample counts.

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
- Transformer backends can enable a mixture-of-experts FFN via the `model.encoder.moe` config block (set `enabled: true` and tune `num_experts`, `top_k`, `capacity_factor`, `ff_multiplier`, `dropout`, and `aux_loss_weight`).

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
## pytorch_optimizer Reference Tables

The tables below enumerate every optimizer, learning-rate scheduler, and loss function exposed by `pytorch-optimizer` v3.8.0. Use these names in the YAML configuration or the `--optimizer`, `--lr_scheduler`, and `--loss` CLI flags.

### Optimizers

| Name | Name | Name | Name |
| --- | --- | --- | --- |
| a2grad | aggmo | galore | rmsprop |
| accsgd | aida | grams | scalableshampoo |
| adabelief | alice | gravity | schedulefreeadamw |
| adabound | alig | grokfastadamw | schedulefreeradam |
| adadelta | amos | kate | schedulefreesgd |
| adafactor | apollo | kron | scion |
| adagc | apollodqn | lamb | scionlight |
| adahessian | asgd | laprop | sgd |
| adai | avagrad | lars | sgdp |
| adalite | bsam | lbfgs | sgdsai |
| adalomo | came | lion | sgdw |
| adam | dadaptadagrad | lomo | shampoo |
| adamax | dadaptadam | madgrad | signsgd |
| adamc | dadaptadan | mars | simplifiedademamix |
| adamg | dadaptlion | msvag | sm3 |
| adammini | dadaptsgd | muon | soap |
| adamod | demo | nadam | sophiah |
| adamp | diffgrad | nero | spam |
| adams | distributedmuon | novograd | splus |
| adamuon | emofact | padam | srmm |
| adamw | emolynx | pid | stableadamw |
| adamwsn | emonavi | pnm | stablespam |
| adan | emoneco | prodigy | swats |
| adanorm | emozeal | qhadam | tam |
| adapnm | exadam | qhm | tiger |
| adashift | fadam | racs | vsgd |
| adasmooth | fira | radam | yogi |
| adatam | focus | ranger |  |
| ademamix | fromage | ranger21 |  |
| adopt | ftrl | ranger25 |  |

### LR Schedulers

| Name | Name | Name | Name |
| --- | --- | --- | --- |
| chebyshev | cosine_annealing_with_warm_restart | multi_step | proportion |
| constant | cosine_annealing_with_warmup | multiplicative | rex |
| cosine | cyclic | one_cycle | step |
| cosine_annealing | linear | poly | warmup_stable_decay |

### Loss Functions

| Name | Name | Name |
| --- | --- | --- |
| bcefocalloss | focalcosineloss | lovaszhingeloss |
| bceloss | focalloss | softf1loss |
| binarybitemperedlogisticloss | focaltverskyloss | tverskyloss |
| bitemperedlogisticloss | jaccardloss |  |
| diceloss | ldamloss |  |

