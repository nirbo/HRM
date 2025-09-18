# trainer with advanced logging and checkpointing
import contextlib
import json
import math
import multiprocessing as mp  # Provide worker pools for parallel dataset parsing
import os
from array import array  # Store JSONL byte offsets compactly when streaming large datasets
import random
import re
import time
import shutil
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import warnings
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from rich.console import Console

from hrm_lm.data.synthetic import build_synthetic_dataset, pad_batch
from hrm_lm.models.hybrid import HRMLanguageModel

warnings.filterwarnings('ignore', message='.*Nested Tensor.*')
console = Console(highlight=False)


def set_seed(seed: int) -> None:
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


def make_model(cfg) -> HRMLanguageModel:
  model = HRMLanguageModel(
    vocab_size=cfg.model.vocab_size,
    d_model=cfg.model.d_model,
    enc_layers=cfg.model.encoder.n_layers,
    dec_layers=cfg.model.decoder.n_layers,
    max_enc_len=cfg.model.encoder.max_seq_len,
    max_dec_len=cfg.model.decoder.max_seq_len,
    hrm_cfg=dict(cfg.model.hrm),
    bridge_cfg=dict(cfg.bridge),
    enc_backend=cfg.model.encoder.backend,
  )
  return model


def make_optimizer(name: str, model: nn.Module, cfg, lr: float):
  betas = tuple(cfg.optim.betas)
  weight_decay = cfg.optim.weight_decay
  name = name.lower()
  if name == 'adamw':
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
  if name == 'adamw_8bit':
    try:
      import bitsandbytes as bnb  # type: ignore
    except ImportError as exc:
      raise ImportError('bitsandbytes is required for --optimizer adamw_8bit. Install via `pip install bitsandbytes`.') from exc
    return bnb.optim.AdamW8bit(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
  raise ValueError(f"Unsupported optimizer '{name}'. Choose from ['adamw', 'adamw_8bit'].")


def demo_batch(cfg, device: torch.device, batch_size: int):
  seq_len = cfg.train.seq_len
  tgt_len = cfg.train.tgt_len
  x = torch.randint(0, cfg.model.vocab_size, (batch_size, seq_len), device=device)
  y_in = torch.randint(0, cfg.model.vocab_size, (batch_size, tgt_len), device=device)
  y = torch.randint(0, cfg.model.vocab_size, (batch_size, tgt_len), device=device)
  return x, y_in, y


def load_jsonl_dataset(directory: Path, workers: int = 1):
  cache_threshold = 250000  # Cap in-memory caching at 250k samples before switching to streaming mode.
  safe_workers = max(1, min(workers, mp.cpu_count() or 1))  # Clamp worker count to the local CPU capacity and ensure at least one process.
  train_file = directory / 'train.jsonl'  # Locate the training split JSONL file.
  val_file = directory / 'val.jsonl'  # Locate the validation split JSONL file.
  meta_file = directory / 'meta.json'  # Locate optional metadata describing padding/tokenizer settings.
  if not train_file.exists():  # Verify the training split exists before proceeding.
    raise ValueError(f"train.jsonl not found in {directory}")  # Abort with a descriptive error when the split is missing.
  if not val_file.exists():  # Verify the validation split exists before proceeding.
    raise ValueError(f"val.jsonl not found in {directory}")  # Abort with a descriptive error when the split is missing.

  pad_id = 0  # Default padding token id when metadata does not specify one.
  vocab_override = None  # Placeholder for tokenizer vocabulary size override pulled from metadata.
  tokenizer_path = None  # Placeholder for tokenizer artifact path harvested from metadata.
  if meta_file.exists():  # Attempt to load metadata when the optional file is present.
    with meta_file.open('r', encoding='utf-8') as meta_handle:  # Open meta.json using UTF-8 decoding.
      meta = json.load(meta_handle)  # Deserialize the metadata payload into a dictionary.
      pad_id = int(meta.get('pad_id', 0))  # Capture the padding token id if present.
      vocab_override = meta.get('vocab_size')  # Capture the tokenizer vocabulary size when present.
      tokenizer_path = meta.get('tokenizer_file')  # Capture the tokenizer artifact location when present.

  def build_split(path: Path, label: str):  # Construct either an in-memory cache or streaming split for the given JSONL file.
    offsets = _index_jsonl(path)  # Record byte offsets for every JSONL record to enable random access while streaming.
    sample_count = len(offsets)  # Determine how many samples are available in the split.
    if sample_count <= cache_threshold:  # Prefer in-memory caching for small datasets to maximize throughput.
      data = _materialize_split(path, label, safe_workers)  # Parse the entire split into memory using the requested worker count.
      return data, sample_count  # Hand back the materialized dataset along with its sample count.
    console.print(f'[grey58]Streaming {label} split ({sample_count:,} samples) without caching.[/grey58]')  # Inform the user that streaming mode will be used.
    return JSONLSplit(path, pad_id, offsets, heartbeat=200000), sample_count  # Build a streaming split that loads samples on demand using recorded offsets.

  train_data, train_count = build_split(train_file, 'train')  # Prepare the training split and capture its sample count.
  val_data, val_count = build_split(val_file, 'val')  # Prepare the validation split and capture its sample count.
  if not train_count:  # Ensure the training split produced at least one usable sample.
    raise ValueError(f'No training samples found in {train_file}')  # Abort when the training split is empty.
  if not val_count:  # Ensure the validation split produced at least one usable sample.
    raise ValueError(f'No validation samples found in {val_file}')  # Abort when the validation split is empty.
  return train_data, val_data, pad_id, vocab_override, tokenizer_path, train_count  # Return dataset payloads plus metadata for downstream consumption.



def dataset_iterator(samples, batch_size: int, pad_id: int, shuffle: bool) -> Iterator:
  if isinstance(samples, JSONLSplit):  # Detect streaming datasets backed by JSON line offsets.
    offsets = samples.offsets  # Access the recorded byte offsets for random access into the JSONL file.
    heartbeat = samples.heartbeat  # Reuse the configured heartbeat interval for progress updates.
    while True:  # Produce batches indefinitely for the training loop.
      if shuffle:  # Shuffle offsets each epoch when randomness is requested.
        order = list(range(len(offsets)))  # Build an index list referencing each offset.
        random.shuffle(order)  # Shuffle the index list in place to randomize sampling order.
      else:
        order = range(len(offsets))  # Preserve deterministic ordering for evaluation when shuffle is disabled.
      emitted = 0  # Track how many samples have been yielded in the current epoch.
      next_ping = heartbeat if heartbeat > 0 else None  # Initialize the next heartbeat threshold when enabled.
      with samples.path.open('rb') as handle:  # Open the JSONL file in binary mode for efficient seeks.
        total = len(order)  # Determine how many positions will be iterated in this epoch.
        for start_idx in range(0, total, batch_size):  # Iterate over batches of index positions.
          batch_indices = order[start_idx:start_idx + batch_size]  # Select the slice of indices forming the next batch.
          batch_samples = []  # Accumulate decoded samples for the batch.
          for idx in batch_indices:  # Iterate over each index within the batch.
            handle.seek(offsets[idx])  # Jump to the byte offset for the requested sample.
            line = handle.readline()  # Read the JSONL record at the current offset.
            parsed = _parse_json_line(line)  # Decode the JSON payload into token triples.
            if parsed is None:  # Skip malformed records that fail validation.
              continue  # Skip appending the malformed sample and continue processing the batch.
            batch_samples.append(parsed)  # Append the decoded sample to the batch list.
          if not batch_samples:  # Guard against empty batches when every record in the slice was malformed.
            continue  # Skip yielding and move on to the next slice.
          emitted += len(batch_samples)  # Update the emitted sample counter for heartbeat reporting.
          if next_ping is not None and emitted >= next_ping:  # Emit a streaming heartbeat when the threshold is met.
            console.print(f'[grey58]streaming samples processed: {emitted:,}[/grey58]', soft_wrap=False, overflow='crop')  # Provide real-time feedback during streaming loads.
            next_ping += heartbeat  # Schedule the next heartbeat threshold for subsequent batches.
          yield pad_batch(batch_samples, pad_id=pad_id)  # Pad sequences and yield the batch to the caller.
  else:  # Fall back to the legacy in-memory iterator for cached datasets.
    while True:
      if shuffle:
        random.shuffle(samples)
      for start in range(0, len(samples), batch_size):
        batch_samples = samples[start:start + batch_size]
        if not batch_samples:
          continue
        yield pad_batch(list(batch_samples), pad_id=pad_id)



def _parse_json_line(line):
  if isinstance(line, bytes):  # Decode raw bytes when the caller streams from a binary file handle.
    line = line.decode('utf-8')  # Convert bytes to a UTF-8 string prior to JSON parsing.
  sample = json.loads(line)  # Decode the raw JSON payload into a Python dictionary.
  enc = sample.get('encoder_ids') or sample.get('input_ids')  # Retrieve encoder token ids from supported keys.
  dec = sample.get('decoder_input_ids') or enc  # Retrieve decoder input ids, defaulting to encoder ids when absent.
  labels = sample.get('labels') or sample.get('targets')  # Retrieve supervision labels from supported keys.
  if enc is None or dec is None or labels is None:  # Skip malformed entries missing the required tensors.
    return None  # Signal to the caller that this line should be ignored.
  return enc, dec, labels  # Return the decoded sample triple for downstream batching.


def _parse_jsonl_chunk(lines: List[str]):
  parsed = []  # Accumulate decoded samples drawn from the provided chunk.
  for line in lines:  # Iterate over each serialized record inside the chunk.
    sample = _parse_json_line(line)  # Decode the JSON payload for the current record.
    if sample is None:  # Skip records that fail validation.
      continue  # Continue with the next record in the chunk.
    parsed.append(sample)  # Append valid samples to the chunk buffer.
  return parsed  # Return all valid samples extracted from this chunk.


def _materialize_split(path: Path, label: str, workers: int):
  samples = []  # Accumulate decoded samples for in-memory caching.
  heartbeat = 200000  # Emit progress every 200k samples to reassure the user during ingestion.
  next_ping = heartbeat  # Initialize the next heartbeat threshold.
  if workers <= 1:  # Use sequential loading when no additional workers are requested.
    with console.status(f'[grey50]loading {label} split from {path}...[/grey50]', spinner='dots'):  # Display a spinner while parsing the file.
      with path.open('r', encoding='utf-8') as handle:  # Open the JSONL file using UTF-8 decoding.
        for line in handle:  # Iterate over each record in the split.
          parsed = _parse_json_line(line)  # Decode the JSON payload into token triples.
          if parsed is None:  # Skip malformed records that fail validation.
            continue  # Move on to the next record without updating counters.
          samples.append(parsed)  # Append the decoded triple to the in-memory cache.
          if len(samples) >= next_ping:  # Check whether it is time to emit a heartbeat message.
            console.print(f'[grey58]{label} samples processed: {len(samples):,}[/grey58]', soft_wrap=False, overflow='crop')  # Provide visibility into loader progress.
            next_ping += heartbeat  # Schedule the next heartbeat threshold.
  else:  # Engage multiprocessing to accelerate parsing for moderate-sized datasets.
    chunk_size = 8192  # Dispatch JSON lines to workers in manageable blocks.
    with console.status(f'[grey50]loading {label} split from {path} with {workers} workers...[/grey50]', spinner='dots'):  # Display spinner with worker count context.
      with path.open('r', encoding='utf-8') as handle:  # Open the JSONL file for streaming.
        def chunk_generator():  # Produce batches of lines to feed into the worker pool.
          chunk: List[str] = []  # Initialize the current chunk buffer.
          for line in handle:  # Stream each record from the file.
            chunk.append(line)  # Append the record to the active chunk.
            if len(chunk) >= chunk_size:  # Emit the chunk once it reaches the configured size.
              yield chunk  # Yield the populated chunk to the caller.
              chunk = []  # Reset the chunk buffer for subsequent records.
          if chunk:  # Emit any trailing records that did not fill a complete chunk.
            yield chunk  # Yield the final partial chunk to the caller.

        with mp.Pool(processes=workers) as pool:  # Launch a worker pool for parallel decoding.
          for parsed_chunk in pool.imap(_parse_jsonl_chunk, chunk_generator(), chunksize=1):  # Stream decoded results back in order.
            if not parsed_chunk:  # Skip empty batches returned by the workers.
              continue  # Continue retrieving additional chunks.
            samples.extend(parsed_chunk)  # Append decoded samples to the in-memory cache.
            if len(samples) >= next_ping:  # Emit progress once the heartbeat threshold is met.
              console.print(f'[grey58]{label} samples processed: {len(samples):,}[/grey58]', soft_wrap=False, overflow='crop')  # Provide visibility into loader progress.
              next_ping += heartbeat  # Schedule the next heartbeat threshold.
  console.print(f'[grey70]{label} split loaded: {len(samples):,} samples[/grey70]', soft_wrap=False, overflow='crop')  # Summarize the final in-memory sample count.
  return samples  # Return the fully materialized dataset.


def _index_jsonl(path: Path):
  offsets = array('Q')  # Store byte offsets compactly using an unsigned 64-bit array.
  heartbeat = 200000  # Emit progress every 200k samples while indexing.
  next_ping = heartbeat  # Initialize the first heartbeat threshold.
  with console.status(f'[grey50]indexing offsets for {path}...[/grey50]', spinner='dots'):  # Display progress while scanning the file.
    with path.open('rb') as handle:  # Open the JSONL file in binary mode to obtain accurate byte positions.
      position = handle.tell()  # Track the current byte offset before reading lines.
      count = 0  # Track how many records have been indexed so far.
      for line in handle:  # Iterate over each raw record in the file.
        offsets.append(position)  # Record the starting byte offset for the current record.
        position = handle.tell()  # Capture the starting byte offset for the next record.
        count += 1  # Increment the indexed record counter.
        if count >= next_ping:  # Emit a heartbeat after processing the configured number of records.
          console.print(f'[grey58]indexed {count:,} samples from {path}[/grey58]', soft_wrap=False, overflow='crop')  # Provide visibility into indexing progress.
          next_ping += heartbeat  # Schedule the next heartbeat threshold.
  console.print(f'[grey70]indexed {len(offsets):,} samples from {path}[/grey70]', soft_wrap=False, overflow='crop')  # Summarize the final index size.
  return offsets  # Return the array of byte offsets for downstream streaming.


class JSONLSplit:
  def __init__(self, path: Path, pad_id: int, offsets: array, heartbeat: int = 200000):
    self.path = path  # Persist the JSONL file path for subsequent streaming reads.
    self.pad_id = pad_id  # Remember the padding token id associated with this split.
    self.offsets = offsets  # Store byte offsets enabling random access into the JSONL file.
    self.heartbeat = heartbeat  # Record the heartbeat interval for progress updates.

  def __len__(self):
    return len(self.offsets)  # Expose the number of available samples in the split.

STEP_DIR_PATTERN = re.compile(r'^step_(\d+)$')


def list_step_checkpoints(directory: Path) -> List[Tuple[int, Path]]:
  ckpts: List[Tuple[int, Path]] = []
  for path in directory.iterdir():
    if not path.is_dir():
      continue
    match = STEP_DIR_PATTERN.match(path.name)
    if not match:
      continue
    model_path = path / 'model.pt'
    if not model_path.exists():
      continue
    step = int(match.group(1))
    ckpts.append((step, path))
  ckpts.sort(key=lambda item: item[0])
  return ckpts


def find_latest_checkpoint(directory: Path, device: torch.device) -> Optional[Tuple[int, Path, dict]]:
  candidates: List[Tuple[int, Path]] = list_step_checkpoints(directory)
  final_dir = directory / 'final'
  final_model = final_dir / 'model.pt'
  if final_model.exists():
    try:
      data = torch.load(final_model, map_location=device)
      candidates.append((int(data.get('step', 0)), final_dir))
    except Exception:
      pass
  if not candidates:
    return None
  candidates.sort(key=lambda item: item[0])
  step, path = candidates[-1]
  model_path = path / 'model.pt'
  data = torch.load(model_path, map_location=device)
  return step, path, data


def enforce_checkpoint_limit(directory: Path, limit: int) -> None:
  if limit <= 0:
    return
  ckpts = list_step_checkpoints(directory)
  while len(ckpts) > limit:
    _, path = ckpts.pop(0)
    try:
      shutil.rmtree(path)
    except FileNotFoundError:
      pass




def write_config(path: Path, cfg_serializable) -> None:
  try:
    cfg_node = OmegaConf.create(cfg_serializable)
    OmegaConf.save(cfg_node, path)
  except Exception as exc:
    console.print(f'[bold red]Failed to write config {path}: {exc}[/bold red]')
def ensure_checkpoint_artifacts(directory: Optional[Path], artifacts: List[Tuple[Path, str]]) -> None:
  if directory is None:
    return
  for source, name in artifacts:
    try:
      if not source.exists():
        continue
      target = directory / name
      if target.exists():
        continue
      shutil.copy2(source, target)
    except Exception as exc:
      console.print(f'[bold red]Failed to copy {source} to {directory}: {exc}[/bold red]')

def build_checkpoint_payload(model, optimizer, scaler, cfg_serializable, step: int, best_loss: float, val_loss: Optional[float], optimizer_name: str) -> dict:
  payload = {
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'cfg': cfg_serializable,
    'step': step,
    'best_loss': best_loss,
    'optimizer_name': optimizer_name,
  }
  if val_loss is not None:
    payload['val_loss'] = float(val_loss)
  if scaler is not None and scaler.is_enabled():
    payload['scaler'] = scaler.state_dict()
  return payload


def gradient_norm(model: nn.Module) -> float:
  total = 0.0
  for param in model.parameters():
    if param.grad is None:
      continue
    grad = param.grad.data
    total += float(torch.sum(grad * grad))
  return math.sqrt(total) if total > 0 else 0.0


def format_eta(seconds: float) -> str:
  if math.isinf(seconds) or seconds <= 0:
    return '--h:--m'
  minutes, _ = divmod(int(seconds + 0.5), 60)
  hours, minutes = divmod(minutes, 60)
  return f"{hours:02d}h:{minutes:02d}m"


def format_speed(seconds_per_it: float) -> str:
  if seconds_per_it <= 0:
    return 'inf it/s'
  if seconds_per_it >= 1.0:
    return f"{seconds_per_it:.2f}s/it"
  else:
    return f"{1.0 / seconds_per_it:.2f}it/s"


def set_learning_rate(optimizer, lr: float) -> None:
  for group in optimizer.param_groups:
    group['lr'] = lr


def main():
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--config', default=None)
  parser.add_argument('--dry_run', action='store_true')
  parser.add_argument('--dataset', default=None)
  parser.add_argument('--batch_size', type=int, default=None)
  parser.add_argument('--eval_batch_size', type=int, default=None)  # optional override for validation batch sizing
  parser.add_argument('--optimizer', default='adamw', choices=['adamw', 'adamw_8bit'])
  parser.add_argument('--learning_rate', type=float, default=None)
  parser.add_argument('--warmup_steps', type=int, default=0)
  parser.add_argument('--lr_scheduler', default='cosine', choices=['cosine', 'linear', 'constant'])  # selects LR decay scheme
  parser.add_argument('--steps', type=int, default=0)
  parser.add_argument('--epochs', type=int, default=0)
  parser.add_argument('--val_every', type=int, default=0)
  parser.add_argument('--save_dir', default=None)
  parser.add_argument('--mixed_precision', default='none')
  parser.add_argument('--grad_clip', type=float, default=0.0)
  parser.add_argument('--checkpoint_limit', type=int, default=0)
  parser.add_argument('--run_name', default=None)
  parser.add_argument('--save_best_model', action='store_true')
  parser.add_argument('--max_seq_len', type=int, default=None)
  parser.add_argument('--log_steps', type=int, default=10)
  parser.add_argument('--dataset_workers', type=int, default=0, help='Number of worker processes for JSONL loading (0 = single process).')  # Allow callers to opt into multiprocessing during dataset ingest.
  args = parser.parse_args()

  cfg = OmegaConf.load(args.config) if args.config else OmegaConf.load(Path(__file__).parent.parent / 'configs' / 'default.yaml')
  requested_workers = args.dataset_workers if args.dataset_workers and args.dataset_workers > 0 else 1  # Capture the requested worker count, defaulting to one process.
  cpu_cap = mp.cpu_count() or 1  # Discover the local CPU capacity to prevent oversubscription.
  dataset_workers = max(1, min(requested_workers, cpu_cap))  # Clamp worker usage within the valid CPU range.
  set_seed(cfg.train.seed)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  effective_batch_size = args.batch_size if args.batch_size is not None else cfg.train.batch_size
  if effective_batch_size <= 0:
    raise ValueError('batch size must be positive')
  cfg.train.batch_size = effective_batch_size

  eval_batch_cfg = getattr(cfg.train, 'eval_batch_size', None)  # pull optional eval batch size from config when present
  eval_batch_candidate = args.eval_batch_size if args.eval_batch_size is not None else eval_batch_cfg  # prefer CLI override when provided
  effective_eval_batch_size = effective_batch_size if eval_batch_candidate is None else int(eval_batch_candidate)  # fallback to train batch size when unset
  if effective_eval_batch_size <= 0:  # ensure evaluation batch size remains valid
    raise ValueError('evaluation batch size must be positive')  # raise explicit error for invalid configuration
  cfg.train.eval_batch_size = effective_eval_batch_size  # persist resolved evaluation batch size in config

  effective_seq_len = args.max_seq_len if args.max_seq_len is not None else cfg.train.seq_len
  if effective_seq_len <= 0:
    raise ValueError('max sequence length must be positive')
  cfg.train.seq_len = effective_seq_len
  cfg.train.tgt_len = effective_seq_len

  base_lr = args.learning_rate if args.learning_rate is not None else cfg.optim.lr
  if base_lr <= 0:
    raise ValueError('learning rate must be positive')

  pad_id = 0
  dataset_size = 0
  artifact_sources: List[Tuple[Path, str]] = []

  if not args.dataset or args.dataset == 'synthetic':
    tokenizer, dataset = build_synthetic_dataset(n=2000, seed=cfg.train.seed)
    dataset_size = len(dataset)

    def data_iter(batch_size: int):  # create a synthetic iterator parameterized by batch size
      while True:  # keep yielding batches indefinitely for streaming training/eval
        batch = [random.choice(dataset) for _ in range(batch_size)]  # sample synthetic examples with replacement
        yield pad_batch(batch)  # pad sequences to shared length

    iterator = data_iter(effective_batch_size)  # training iterator uses resolved train batch size
    val_iterator = data_iter(effective_eval_batch_size)  # evaluation iterator uses resolved eval batch size
  else:
    dataset_path = Path(args.dataset)
    if not dataset_path.exists() or not dataset_path.is_dir():
      raise ValueError('dataset must be "synthetic" or a directory containing train.jsonl/val.jsonl')
    train_data, val_data, pad_id, vocab_override, tokenizer_path, dataset_size = load_jsonl_dataset(dataset_path, workers=dataset_workers)  # Load dataset splits using the configured worker pool.
    if vocab_override is not None and int(vocab_override) > cfg.model.vocab_size:
      cfg.model.vocab_size = int(vocab_override)
    if dataset_size == 0:
      raise ValueError(f'No training samples found in {dataset_path}')
    mode_desc = 'streaming' if isinstance(train_data, JSONLSplit) else 'cached'  # Describe whether the dataset is streamed or fully cached.
    console.print(f'[grey70]Loaded {dataset_size:,} training samples from {dataset_path} ({mode_desc})[/grey70]')
    iterator = dataset_iterator(train_data, effective_batch_size, pad_id=pad_id, shuffle=True)
    val_iterator = dataset_iterator(val_data, effective_eval_batch_size, pad_id=pad_id, shuffle=False)  # evaluation iterator uses dedicated batch size

    meta_path = dataset_path / 'meta.json'
    if meta_path.exists():
      artifact_sources.append((meta_path.resolve(), meta_path.name))
    if tokenizer_path:
      tok_path = Path(tokenizer_path)
      if not tok_path.is_absolute():
        tok_path = (dataset_path / tok_path).resolve()
      if tok_path.exists():
        artifact_sources.append((tok_path, tok_path.name))

  model = make_model(cfg).to(device)
  optimizer = make_optimizer(args.optimizer, model, cfg, base_lr)

  if args.dry_run:
    x, y_in, y = demo_batch(cfg, device, effective_batch_size)
    out = model(x, y_in, labels=y)
    console.print(f'[chartreuse4]dry_run loss:[/chartreuse4] {out['loss'].item():.6f}')
    return

  save_dir: Optional[Path] = None
  best_dir: Optional[Path] = None

  if args.run_name:
    base_dir = Path('runs') / args.run_name
    save_dir = base_dir / 'checkpoints'
    save_dir.mkdir(parents=True, exist_ok=True)
    ensure_checkpoint_artifacts(save_dir, artifact_sources)
    if args.save_best_model:
      best_dir = base_dir / 'best-model'
      best_dir.mkdir(parents=True, exist_ok=True)
      ensure_checkpoint_artifacts(best_dir, artifact_sources)
  elif args.save_dir:
    if args.save_best_model:
      raise ValueError('--save_best_model requires --run_name to place artifacts under runs/<run-name>/best-model/.')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ensure_checkpoint_artifacts(save_dir, artifact_sources)
  else:
    if args.save_best_model:
      raise ValueError('--save_best_model requires --run_name to place artifacts under runs/<run-name>/best-model/.')

  total_steps = args.steps if args.steps > 0 else 0
  if total_steps <= 0:
    if args.epochs <= 0:
      raise ValueError('Specify a positive --steps or --epochs')
    steps_per_epoch = max(1, math.ceil(dataset_size / effective_batch_size))
    total_steps = steps_per_epoch * args.epochs

  mp_mode = args.mixed_precision.lower()
  use_autocast = mp_mode in ('fp16', 'bf16')
  autocast_dtype = torch.float16 if mp_mode == 'fp16' else (torch.bfloat16 if mp_mode == 'bf16' else None)
  fp16_enabled = mp_mode == 'fp16' and device.type == 'cuda'
  scaler = torch.amp.GradScaler('cuda', enabled=fp16_enabled)
  autocast_kwargs = {'device_type': device.type, 'dtype': autocast_dtype} if (use_autocast and autocast_dtype is not None) else None

  cfg_serializable = OmegaConf.to_container(cfg, resolve=True)
  model.train()

  start_step = 0
  best_loss = float('inf')

  if save_dir is not None:
    resume = find_latest_checkpoint(save_dir, device)
    if resume is not None:
      resume_step, resume_path, data = resume
      model.load_state_dict(data['state_dict'])
      if 'optimizer' in data:
        optimizer.load_state_dict(data['optimizer'])
      if scaler is not None and scaler.is_enabled() and data.get('scaler') is not None:
        scaler.load_state_dict(data['scaler'])
      best_loss = float(data.get('best_loss', float('inf')))
      start_step = int(data.get('step', resume_step))
      console.print(f'[bold yellow]Resuming from {resume_path} (step {start_step})[/bold yellow]')
      if start_step >= total_steps:
        console.print(f'[bold green]All {total_steps} steps already completed; exiting.[/bold green]')
        return
    else:
      console.print('[bold cyan]No checkpoint found; starting fresh.[/bold cyan]')

  warmup_steps = max(0, args.warmup_steps)
  log_steps = max(1, args.log_steps)
  run_start_step = start_step
  run_start_time = time.time()

  def adjust_lr(global_step: int) -> float:
    if warmup_steps > 0 and global_step <= warmup_steps:  # apply warmup ramp if configured
      warmup_ratio = min(global_step / warmup_steps, 1.0)  # clamp warmup progress between 0 and 1
      lr = base_lr * warmup_ratio  # scale base LR during warmup
    else:
      decay_steps = max(total_steps - warmup_steps, 1)  # ensure at least one decay step
      decay_progress_raw = (global_step - warmup_steps) / decay_steps  # compute raw decay progress
      decay_progress = min(max(decay_progress_raw, 0.0), 1.0)  # clamp decay progress between 0 and 1
      if args.lr_scheduler == 'cosine':  # select cosine decay multiplier
        decay_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))  # compute cosine multiplier
      elif args.lr_scheduler == 'linear':  # select linear decay multiplier
        decay_factor = 1.0 - decay_progress  # compute linear multiplier
      elif args.lr_scheduler == 'constant':  # keep learning rate constant after warmup
        decay_factor = 1.0  # no decay applied
      else:  # safeguard against unexpected scheduler names
        raise ValueError(f"Unsupported lr scheduler '{args.lr_scheduler}'")  # raise explicit configuration error
      lr = base_lr * decay_factor  # scale base LR by selected decay factor
    set_learning_rate(optimizer, lr)  # push updated LR into optimizer parameter groups
    return lr  # expose the learning rate for logging

  def truncate_to_max_length(tensor: torch.Tensor, length: int) -> torch.Tensor:
    return tensor[:, :length] if tensor.size(1) > length else tensor

  for step in range(start_step, total_steps):
    global_step = step + 1
    current_lr = adjust_lr(global_step)

    enc, dec_in, labels, enc_mask, dec_mask = next(iterator)
    enc = truncate_to_max_length(enc, effective_seq_len).to(device)
    dec_in = truncate_to_max_length(dec_in, effective_seq_len).to(device)
    labels = truncate_to_max_length(labels, effective_seq_len).to(device)
    enc_mask = truncate_to_max_length(enc_mask, effective_seq_len).to(device).bool()
    dec_mask = truncate_to_max_length(dec_mask, effective_seq_len).to(device).bool()

    optimizer.zero_grad(set_to_none=True)
    ctx = torch.autocast(**autocast_kwargs) if autocast_kwargs else contextlib.nullcontext()
    with ctx:
      out = model(enc, dec_in, enc_attn_mask=enc_mask, dec_attn_mask=dec_mask, labels=labels)
      loss = out['loss']

    if scaler is not None and scaler.is_enabled():
      scaler.scale(loss).backward()
      scaler.unscale_(optimizer)
    else:
      loss.backward()

    if args.grad_clip > 0:
      torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

    grad_norm = gradient_norm(model)

    if scaler is not None and scaler.is_enabled():
      scaler.step(optimizer)
      scaler.update()
    else:
      optimizer.step()

    steps_completed = global_step - run_start_step
    elapsed = time.time() - run_start_time
    time_per_step = elapsed / steps_completed if steps_completed > 0 else float('inf')
    eta = format_eta(time_per_step * (total_steps - global_step))
    speed = format_speed(time_per_step)

    if global_step % log_steps == 0 or global_step == total_steps:
      parts = [
        f'[grey70]step {global_step}/{total_steps}[/grey70]',
        f'[chartreuse4]loss {loss.item():.15f}[/chartreuse4]',
        f'[steel_blue]grad {grad_norm:.15f}[/steel_blue]',
        f'[dark_orange3]lr {current_lr:.15f}[/dark_orange3]',
        f'[orchid]eta {eta}[/orchid]',
        f'[medium_spring_green]{speed}[/medium_spring_green]',
      ]
      console.print(' | '.join(parts), soft_wrap=False, overflow='crop')

    if args.val_every > 0 and global_step % args.val_every == 0:
      model.eval()
      with console.status(f'[grey58]eval step {global_step}/{total_steps}...[/grey58]', spinner='line'):
        with torch.no_grad():
          v_enc, v_dec_in, v_labels, v_enc_mask, v_dec_mask = next(val_iterator)
          v_enc = truncate_to_max_length(v_enc, effective_seq_len).to(device)
          v_dec_in = truncate_to_max_length(v_dec_in, effective_seq_len).to(device)
          v_labels = truncate_to_max_length(v_labels, effective_seq_len).to(device)
          v_enc_mask = truncate_to_max_length(v_enc_mask, effective_seq_len).to(device).bool()
          v_dec_mask = truncate_to_max_length(v_dec_mask, effective_seq_len).to(device).bool()
          v_out = model(v_enc, v_dec_in, enc_attn_mask=v_enc_mask, dec_attn_mask=v_dec_mask, labels=v_labels)
          v_loss = float(v_out['loss'].item())
      console.print(f'[orchid]eval:[/orchid] step {global_step}/{total_steps}, loss: {v_loss:.6f}')
      model.train()

      if args.save_best_model and v_loss < best_loss:
        best_loss = v_loss
        if best_dir is not None:
          best_payload = build_checkpoint_payload(model, optimizer, scaler, cfg_serializable, global_step, best_loss, v_loss, args.optimizer)
          best_model_path = best_dir / 'model.pt'
          torch.save(best_payload, best_model_path)
          write_config(best_dir / 'config.yaml', cfg_serializable)
          ensure_checkpoint_artifacts(best_dir, artifact_sources)

      if save_dir is not None:
        payload = build_checkpoint_payload(model, optimizer, scaler, cfg_serializable, global_step, best_loss, v_loss, args.optimizer)
        ckpt_dir = save_dir / f'step_{global_step:06d}'
        if ckpt_dir.exists():
          shutil.rmtree(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload, ckpt_dir / 'model.pt')
        write_config(ckpt_dir / 'config.yaml', cfg_serializable)
        ensure_checkpoint_artifacts(ckpt_dir, artifact_sources)
        enforce_checkpoint_limit(save_dir, args.checkpoint_limit)

  if save_dir is not None:
    final_payload = build_checkpoint_payload(model, optimizer, scaler, cfg_serializable, total_steps, best_loss, None, args.optimizer)
    final_dir = save_dir / 'final'
    if final_dir.exists():
      shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    torch.save(final_payload, final_dir / 'model.pt')
    write_config(final_dir / 'config.yaml', cfg_serializable)
    ensure_checkpoint_artifacts(final_dir, artifact_sources)

if __name__ == '__main__':
  main()
