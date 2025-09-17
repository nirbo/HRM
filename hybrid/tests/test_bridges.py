# bridge mode smoke tests
import torch  # tensor helper
from omegaconf import OmegaConf  # config helper
from hrm_lm.training.train import make_model  # model builder


def run_once(bridge_type: str) -> None:
  cfg = OmegaConf.create({
    'model': {
      'vocab_size': 512,
      'd_model': 64,
      'encoder': {'backend': 'transformer', 'n_layers': 1, 'max_seq_len': 64},
      'decoder': {'n_layers': 1, 'max_seq_len': 32},
      'hrm': {
        'd_model': 64,
        'h_len': 4,
        'l_len': 8,
        'h_layers': 1,
        'l_layers': 1,
        'h_cycles': 2,
        'l_steps': 2,
        'approx_grad': 'one_step',
        'out_dim': 64,
        'use_halting': False
      }
    },
    'bridge': {'type': bridge_type, 'prefix_len': 4}
  })  # build config
  model = make_model(cfg)  # instantiate model
  x = torch.randint(0, 512, (2, 16))  # encoder ids
  y_in = torch.randint(0, 512, (2, 8))  # decoder inputs
  y = torch.randint(0, 512, (2, 8))  # labels
  out = model(x, y_in, labels=y)  # forward pass
  assert out['logits'].shape[:2] == (2, 8)  # check logits shape


def test_prefix_bridge() -> None:
  run_once('prefix')  # test prefix mode


def test_cross_bridge() -> None:
  run_once('cross_attn')  # test cross attention mode
