# toy trainer with dry-run
import contextlib
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from hrm_lm.data.synthetic import build_synthetic_dataset, pad_batch  # dataset utilities
from hrm_lm.models.hybrid import HRMLanguageModel


def set_seed(seed: int) -> None:
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


def make_model(cfg):
  model = HRMLanguageModel(
    vocab_size=cfg.model.vocab_size,
    d_model=cfg.model.d_model,
    enc_layers=cfg.model.encoder.n_layers,
    dec_layers=cfg.model.decoder.n_layers,
    max_enc_len=cfg.model.encoder.max_seq_len,
    max_dec_len=cfg.model.decoder.max_seq_len,
    hrm_cfg=dict(cfg.model.hrm),
    bridge_cfg=dict(cfg.bridge),
    enc_backend=cfg.model.encoder.backend
  )
  return model


def demo_batch(cfg, device):
  batch = cfg.train.batch_size
  seq_len = cfg.train.seq_len
  tgt_len = cfg.train.tgt_len
  x = torch.randint(0, cfg.model.vocab_size, (batch, seq_len), device=device)
  y_in = torch.randint(0, cfg.model.vocab_size, (batch, tgt_len), device=device)
  y = torch.randint(0, cfg.model.vocab_size, (batch, tgt_len), device=device)
  return x, y_in, y


def list_step_checkpoints(directory: Path) -> List[Tuple[int, Path]]:
  ckpts: List[Tuple[int, Path]] = []
  for path in directory.glob('step_*.pt'):
    name = path.stem.split('_')
    if len(name) != 2:
      continue
    try:
      step = int(name[1])
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
    step, path = ckpts.pop(0)
    try:
      path.unlink()
    except FileNotFoundError:
      pass


def build_checkpoint_payload(model, optimizer, scaler, cfg_serializable, step: int, best_loss: float, val_loss: Optional[float]) -> dict:
  payload = {
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'cfg': cfg_serializable,
    'step': step,
    'best_loss': best_loss,
  }
  if val_loss is not None:
    payload['val_loss'] = float(val_loss)
  if scaler is not None and scaler.is_enabled():
    payload['scaler'] = scaler.state_dict()
  return payload


def main():
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--config', default=None)
  parser.add_argument('--dry_run', type=int, default=1)
  parser.add_argument('--dataset', default=None)
  parser.add_argument('--steps', type=int, default=200)
  parser.add_argument('--val_every', type=int, default=0)
  parser.add_argument('--save_dir', default=None)
  parser.add_argument('--mixed_precision', default='none')
  parser.add_argument('--grad_clip', type=float, default=0.0)
  parser.add_argument('--checkpoint_limit', type=int, default=0)
  parser.add_argument('--run_name', default=None)
  parser.add_argument('--save_best_model', action='store_true')
  args = parser.parse_args()

  cfg = OmegaConf.load(args.config) if args.config else OmegaConf.load(Path(__file__).parent.parent / 'configs' / 'default.yaml')
  set_seed(cfg.train.seed)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = make_model(cfg).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, betas=tuple(cfg.optim.betas), weight_decay=cfg.optim.weight_decay)

  if args.dry_run:
    x, y_in, y = demo_batch(cfg, device)
    out = model(x, y_in, labels=y)
    print('dry_run loss:', out['loss'].item())
    return

  save_dir: Optional[Path] = None
  best_dir: Optional[Path] = None

  if args.run_name:
    base_dir = Path('runs') / args.run_name
    save_dir = base_dir / 'checkpoints'
    save_dir.mkdir(parents=True, exist_ok=True)
    if args.save_best_model:
      best_dir = base_dir / 'best-model'
      best_dir.mkdir(parents=True, exist_ok=True)
  elif args.save_dir:
    if args.save_best_model:
      raise ValueError('--save_best_model requires --run_name to place artifacts under runs/<run-name>/best-model/')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
  else:
    if args.save_best_model:
      raise ValueError('--save_best_model requires --run_name to place artifacts under runs/<run-name>/best-model/')

  if args.dataset == 'synthetic':
    tokenizer, dataset = build_synthetic_dataset(n=2000, seed=cfg.train.seed)

    def data_iter():
      while True:
        batch = [random.choice(dataset) for _ in range(cfg.train.batch_size)]
        yield pad_batch(batch)

    iterator = data_iter()
    val_iterator = data_iter()
  else:
    raise ValueError('dataset required for non-dry run')

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
      print(f'Resuming from {resume_path} (step {start_step})')
      if start_step >= args.steps:
        print(f'All {args.steps} steps already completed; exiting.')
        return

  for step in range(start_step, args.steps):
    enc, dec_in, labels, enc_mask, dec_mask = next(iterator)
    enc = enc.to(device)
    dec_in = dec_in.to(device)
    labels = labels.to(device)
    enc_mask = enc_mask.to(device)
    dec_mask = dec_mask.to(device)

    optimizer.zero_grad(set_to_none=True)
    ctx = torch.autocast(**autocast_kwargs) if autocast_kwargs else contextlib.nullcontext()
    with ctx:
      out = model(enc, dec_in, enc_attn_mask=enc_mask, dec_attn_mask=dec_mask, labels=labels)
      loss = out['loss']

    if scaler is not None and scaler.is_enabled():
      scaler.scale(loss).backward()
      if args.grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
      scaler.step(optimizer)
      scaler.update()
    else:
      loss.backward()
      if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
      optimizer.step()

    global_step = step + 1
    if global_step % 10 == 0:
      print(f'step {global_step} loss {loss.item():.4f}')

    if args.val_every > 0 and global_step % args.val_every == 0:
      model.eval()
      with torch.no_grad():
        v_enc, v_dec_in, v_labels, v_enc_mask, v_dec_mask = next(val_iterator)
        v_enc = v_enc.to(device)
        v_dec_in = v_dec_in.to(device)
        v_labels = v_labels.to(device)
        v_enc_mask = v_enc_mask.to(device)
        v_dec_mask = v_dec_mask.to(device)
        v_out = model(v_enc, v_dec_in, enc_attn_mask=v_enc_mask, dec_attn_mask=v_dec_mask, labels=v_labels)
        v_loss = float(v_out['loss'].item())
      print(f'val step {global_step} loss {v_loss:.4f}')
      model.train()

      if args.save_best_model and v_loss < best_loss:
        best_loss = v_loss
        if best_dir is not None:
          best_payload = build_checkpoint_payload(model, optimizer, scaler, cfg_serializable, global_step, best_loss, v_loss)
          best_path = best_dir / 'best.pt'
          torch.save(best_payload, best_path)

      if save_dir is not None:
        payload = build_checkpoint_payload(model, optimizer, scaler, cfg_serializable, global_step, best_loss, v_loss)
        ckpt_path = save_dir / f'step_{global_step}.pt'
        torch.save(payload, ckpt_path)
        enforce_checkpoint_limit(save_dir, args.checkpoint_limit)

  if save_dir is not None:
    final_payload = build_checkpoint_payload(model, optimizer, scaler, cfg_serializable, args.steps, best_loss, None)
    torch.save(final_payload, save_dir / 'final.pt')

if __name__ == '__main__':
  main()
