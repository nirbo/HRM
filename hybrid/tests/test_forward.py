# minimal shape test
import torch
from omegaconf import OmegaConf
from hrm_lm.training.train import make_model

def test_forward():
  cfg = OmegaConf.create({
    'model': {'vocab_size': 1024, 'd_model': 128, 'encoder': {'backend': 'transformer', 'n_layers': 2, 'max_seq_len': 256}, 'decoder': {'n_layers': 2, 'max_seq_len': 128}, 'hrm': {'d_model': 128, 'h_len': 4, 'l_len': 16, 'h_layers': 1, 'l_layers': 1, 'h_cycles': 2, 'l_steps': 2, 'approx_grad': 'one_step', 'out_dim': 128, 'use_halting': False}}, 'bridge': {'type': 'prefix', 'prefix_len': 4}})
  m = make_model(cfg)
  x = torch.randint(0, 1024, (2, 32))
  y_in = torch.randint(0, 1024, (2, 16))
  y = torch.randint(0, 1024, (2, 16))
  out = m(x, y_in, labels=y)
  assert out['logits'].shape[:2] == (2, 16)
  assert out['loss'] is not None
  print('ok')
