# minimal shape test with deep supervision
import torch  # tensor library
from omegaconf import OmegaConf  # config helper
from hrm_lm.training.train import make_model  # model builder


def test_forward():
  cfg = OmegaConf.create({
    'model': {
      'vocab_size': 1024,
      'd_model': 128,
      'encoder': {'backend': 'transformer', 'n_layers': 2, 'max_seq_len': 256},
      'decoder': {'n_layers': 2, 'max_seq_len': 128},
      'hrm': {
        'd_model': 128,
        'h_len': 4,
        'l_len': 16,
        'h_layers': 1,
        'l_layers': 1,
        'h_cycles': 2,
        'l_steps': 2,
        'approx_grad': 'one_step',
        'out_dim': 128,
        'use_halting': False,
        'deep_supervision': True,
        'ds_weight': 0.1
      }
    },
    'bridge': {'type': 'prefix', 'prefix_len': 4}
  })  # build config with DS
  model = make_model(cfg)  # instantiate model
  x = torch.randint(0, 1024, (2, 32))  # encoder tokens
  y_in = torch.randint(0, 1024, (2, 16))  # decoder inputs
  y = torch.randint(0, 1024, (2, 16))  # labels
  out = model(x, y_in, labels=y)  # forward pass
  assert out['logits'].shape[:2] == (2, 16)  # logits shape check
  assert out['loss'] is not None  # ensure loss computed
