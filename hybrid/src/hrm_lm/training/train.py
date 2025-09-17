# toy trainer with dry-run
import os, math, torch, random
import torch.nn as nn
from omegaconf import OmegaConf
from hrm_lm.models.hybrid import HRMLanguageModel

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
  args = ap.parse_args()

  cfg = OmegaConf.load(args.config) if args.config else OmegaConf.load(os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml'))
  set_seed(cfg.train.seed)

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = make_model(cfg).to(device)
  opt = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, betas=tuple(cfg.optim.betas), weight_decay=cfg.optim.weight_decay)

  if args.dry_run:
    x, y_in, y = demo_batch(cfg, device)
    out = model(x, y_in, labels=y)
    print('dry_run loss:', float(out['loss']))
    return

  # placeholder training loop
  for step in range(10):
    x, y_in, y = demo_batch(cfg, device)
    out = model(x, y_in, labels=y)
    loss = out['loss']
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if step % 1 == 0:
      print(f'step {step} loss {float(loss):.4f}')

if __name__ == '__main__':
  main()
