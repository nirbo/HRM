# trainer with advanced logging and checkpointing
import contextlib
import json
import math
import os
import random
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


def load_jsonl_dataset(directory: Path):
  train_file = directory / 'train.jsonl'
  val_file = directory / 'val.jsonl'
  meta_file = directory / 'meta.json'
  if not train_file.exists():
    raise ValueError(f"train.jsonl not found in {directory}")
  if not val_file.exists():
    raise ValueError(f"val.jsonl not found in {directory}")

  pad_id = 0
  vocab_override = None
  tokenizer_path = None
  if meta_file.exists():
    with meta_file.open('r', encoding='utf-8') as f:
      meta = json.load(f)
      pad_id = int(meta.get('pad_id', 0))
      vocab_override = meta.get('vocab_size')
      tokenizer_path = meta.get('tokenizer_file')

  def read_file(path: Path):
    samples = []
    with path.open('r', encoding='utf-8') as f:
      for line in f:
        sample = json.loads(line)
        enc = sample.get('encoder_ids') or sample.get('input_ids')
        dec = sample.get('decoder_input_ids') or enc
        labels = sample.get('labels') or sample.get('targets')
        if enc is None or dec is None or labels is None:
          continue
        samples.append((enc, dec, labels))
    return samples

  train_data = read_file(train_file)
  val_data = read_file(val_file)
  if not train_data:
    raise ValueError(f'No training samples found in {train_file}')
  if not val_data:
    raise ValueError(f'No validation samples found in {val_file}')
  return train_data, val_data, pad_id, vocab_override, tokenizer_path


def dataset_iterator(samples, batch_size: int, pad_id: int, shuffle: bool) -> Iterator:
  while True:
    if shuffle:
      random.shuffle(samples)
    for start in range(0, len(samples), batch_size):
      batch_samples = samples[start:start + batch_size]
      if not batch_samples:
        continue
      yield pad_batch(list(batch_samples), pad_id=pad_id)


def list_step_checkpoints(directory: Path) -> List[Tuple[int, Path]]:
  ckpts: List[Tuple[int, Path]] = []
  for path in directory.glob('step_*.pt'):
    parts = path.stem.split('_')
    if len(parts) != 2:
      continue
    try:
      step = int(parts[1])
    except ValueError:
      continue
    ckpts.append((step, path))
  ckpts.sort(key=lambda item: item[0])
  return ckpts


def find_latest_checkpoint(directory: Path, device: torch.device) -> Optional[Tuple[int, Path, dict]]:
  candidates: List[Tuple[int, Path]] = list_step_checkpoints(directory)
  final_path = directory / 'final.pt'
  if final_path.exists():
    try:
      data = torch.load(final_path, map_location=device)
      candidates.append((int(data.get('step', 0)), final_path))
    except Exception:
      pass
  if not candidates:
    return None
  candidates.sort(key=lambda item: item[0])
  step, path = candidates[-1]
  data = torch.load(path, map_location=device)
  return step, path, data


def enforce_checkpoint_limit(directory: Path, limit: int) -> None:
  if limit <= 0:
    return
  ckpts = list_step_checkpoints(directory)
  while len(ckpts) > limit:
    _, path = ckpts.pop(0)
    try:
      path.unlink()
    except FileNotFoundError:
      pass


def persist_checkpoint_config(directory: Path, stem: str, cfg_serializable) -> None:
  cfg_path = directory / f'{stem}.yaml'  # derive config file path alongside checkpoint
  try:  # attempt to persist config without interrupting training
    cfg_node = OmegaConf.create(cfg_serializable)  # rebuild OmegaConf node for serialization
    OmegaConf.save(cfg_node, cfg_path)  # write config snapshot to disk
  except Exception as exc:  # capture serialization issues
    console.print(f'[bold red]Failed to write config {cfg_path}: {exc}[/bold red]')  # surface failure without aborting training

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
  args = parser.parse_args()

  cfg = OmegaConf.load(args.config) if args.config else OmegaConf.load(Path(__file__).parent.parent / 'configs' / 'default.yaml')
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
    train_data, val_data, pad_id, vocab_override, tokenizer_path = load_jsonl_dataset(dataset_path)
    if vocab_override is not None and int(vocab_override) > cfg.model.vocab_size:
      cfg.model.vocab_size = int(vocab_override)
    dataset_size = len(train_data)
    if dataset_size == 0:
      raise ValueError(f'No training samples found in {dataset_path}')
    console.print(f'[grey70]Loaded {dataset_size} training samples from {dataset_path}[/grey70]')
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
          torch.save(best_payload, best_dir / 'best.pt')
          persist_checkpoint_config(best_dir, 'best', cfg_serializable)  # persist matching config snapshot for best checkpoint

      if save_dir is not None:
        payload = build_checkpoint_payload(model, optimizer, scaler, cfg_serializable, global_step, best_loss, v_loss, args.optimizer)
        ckpt_path = save_dir / f'step_{global_step}.pt'
        torch.save(payload, ckpt_path)
        persist_checkpoint_config(save_dir, f'step_{global_step}', cfg_serializable)  # persist config alongside step checkpoint
        enforce_checkpoint_limit(save_dir, args.checkpoint_limit)

  if save_dir is not None:
    final_payload = build_checkpoint_payload(model, optimizer, scaler, cfg_serializable, total_steps, best_loss, None, args.optimizer)
    torch.save(final_payload, save_dir / 'final.pt')
    persist_checkpoint_config(save_dir, 'final', cfg_serializable)  # persist config for final snapshot

if __name__ == '__main__':
  main()
