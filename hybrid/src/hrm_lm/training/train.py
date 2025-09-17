# toy trainer with dry-run
import os, math, torch, random
import contextlib
import torch.nn as nn
from omegaconf import OmegaConf
from hrm_lm.models.hybrid import HRMLanguageModel
from hrm_lm.data.synthetic import build_synthetic_dataset, pad_batch  # dataset utilities

def set_seed(s):
  random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def make_model(cfg):
  m = HRMLanguageModel(
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
  return m

def demo_batch(cfg, device):
  B = cfg.train.batch_size
  S = cfg.train.seq_len
  T = cfg.train.tgt_len
  x = torch.randint(0, cfg.model.vocab_size, (B, S), device=device)
  y_in = torch.randint(0, cfg.model.vocab_size, (B, T), device=device)
  y = torch.randint(0, cfg.model.vocab_size, (B, T), device=device)
  return x, y_in, y

def main():
  import argparse
  ap = argparse.ArgumentParser()
  ap.add_argument('--config', default=None)
  ap.add_argument('--dry_run', type=int, default=1)
  ap.add_argument('--dataset', default=None)  # dataset name for training
  ap.add_argument('--steps', type=int, default=200)  # training steps for non-dry run
  ap.add_argument('--val_every', type=int, default=0)  # validation frequency in steps
  ap.add_argument('--save_dir', default=None)  # optional directory for checkpoints
  ap.add_argument('--mixed_precision', default='none')  # mixed precision mode (none|fp16|bf16)
  ap.add_argument('--grad_clip', type=float, default=0.0)  # gradient clipping norm
  args = ap.parse_args()

  cfg = OmegaConf.load(args.config) if args.config else OmegaConf.load(os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml'))
  set_seed(cfg.train.seed)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # select device
  model = make_model(cfg).to(device)
  opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, betas=tuple(cfg.optim.betas), weight_decay=cfg.optim.weight_decay)

  if args.dry_run:
    x, y_in, y = demo_batch(cfg, device)
    out = model(x, y_in, labels=y)
    print('dry_run loss:', out['loss'].item())  # report loss without dtype warning
    return

  if args.dataset == 'synthetic':
    tokenizer, dataset = build_synthetic_dataset(n=2000, seed=cfg.train.seed)  # build synthetic data
    def data_iter():
      while True:
        batch = [random.choice(dataset) for _ in range(cfg.train.batch_size)]  # sample batch
        yield pad_batch(batch)  # yield padded batch
    iterator = data_iter()  # persistent iterator
    val_iterator = data_iter()  # validation iterator
  else:
    raise ValueError('dataset required for non-dry run')  # enforce dataset selection

  mp_mode = args.mixed_precision.lower()  # normalize mixed precision mode
  use_autocast = mp_mode in ('fp16', 'bf16')  # decide if autocast is needed
  autocast_dtype = torch.float16 if mp_mode == 'fp16' else (torch.bfloat16 if mp_mode == 'bf16' else None)  # select dtype
  fp16_enabled = mp_mode == 'fp16' and device.type == 'cuda'  # fp16 active only on CUDA
  scaler = torch.amp.GradScaler('cuda', enabled=fp16_enabled)  # fp16 scaler
  autocast_kwargs = {'device_type': device.type, 'dtype': autocast_dtype} if (use_autocast and autocast_dtype is not None) else None  # autocast args
  if args.save_dir:
    os.makedirs(args.save_dir, exist_ok=True)  # ensure checkpoint dir exists
  cfg_serializable = OmegaConf.to_container(cfg, resolve=True)  # serialize config once
  model.train()  # switch to train mode

  for step in range(args.steps):
    enc, dec_in, labels, enc_mask, dec_mask = next(iterator)  # get next batch
    enc = enc.to(device)  # move encoder ids to device
    dec_in = dec_in.to(device)  # move decoder inputs
    labels = labels.to(device)  # move labels
    enc_mask = enc_mask.to(device)  # move encoder mask
    dec_mask = dec_mask.to(device)  # move decoder mask
    opt.zero_grad(set_to_none=True)  # clear gradients
    ctx = torch.autocast(**autocast_kwargs) if autocast_kwargs else contextlib.nullcontext()  # choose precision context
    with ctx:
      out = model(enc, dec_in, enc_attn_mask=enc_mask, dec_attn_mask=dec_mask, labels=labels)  # forward pass
      loss = out['loss']  # capture loss
    if scaler.is_enabled():
      scaler.scale(loss).backward()  # scaled backward
      if args.grad_clip > 0:
        scaler.unscale_(opt)  # unscale grads
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # clip norms
      scaler.step(opt)  # optimizer step via scaler
      scaler.update()  # update scaler
    else:
      loss.backward()  # backpropagate
      if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # clip norms
      opt.step()  # optimizer step
    if (step + 1) % 10 == 0:
      print(f'step {step + 1} loss {loss.item():.4f}')  # training log
    if args.val_every > 0 and (step + 1) % args.val_every == 0:
      model.eval()  # switch to eval
      with torch.no_grad():
        v_enc, v_dec_in, v_labels, v_enc_mask, v_dec_mask = next(val_iterator)  # validation batch
        v_enc = v_enc.to(device)  # move val encoder ids
        v_dec_in = v_dec_in.to(device)  # move val decoder ids
        v_labels = v_labels.to(device)  # move val labels
        v_enc_mask = v_enc_mask.to(device)  # move val encoder mask
        v_dec_mask = v_dec_mask.to(device)  # move val decoder mask
        v_out = model(v_enc, v_dec_in, enc_attn_mask=v_enc_mask, dec_attn_mask=v_dec_mask, labels=v_labels)  # val forward
        v_loss = float(v_out['loss'].item())  # val loss scalar
      print(f'val step {step + 1} loss {v_loss:.4f}')  # validation log
      model.train()  # back to train mode
      if args.save_dir:
        ckpt_path = os.path.join(args.save_dir, f'step_{step + 1}.pt')  # checkpoint path
        torch.save({'state_dict': model.state_dict(), 'cfg': cfg_serializable, 'step': step + 1}, ckpt_path)  # save checkpoint

  if args.save_dir:
    final_path = os.path.join(args.save_dir, 'final.pt')  # final checkpoint path
    torch.save({'state_dict': model.state_dict(), 'cfg': cfg_serializable, 'step': args.steps}, final_path)  # save final checkpoint

if __name__ == '__main__':
  main()
