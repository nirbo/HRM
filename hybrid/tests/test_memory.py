# memory comparison test
import torch  # tensor helper
import pytest  # test decorator
from omegaconf import OmegaConf  # config helper
from hrm_lm.training.train import make_model  # model builder


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA only')
def test_onestep_uses_less_memory() -> None:
  base = {
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
        'h_cycles': 3,
        'l_steps': 3,
        'approx_grad': 'one_step',
        'out_dim': 128,
        'use_halting': False
      }
    },
    'bridge': {'type': 'prefix', 'prefix_len': 4}
  }  # base config
  cfg = OmegaConf.create(base)  # wrap config
  model_one = make_model(cfg).cuda()  # build one-step model
  x = torch.randint(0, 1024, (2, 32), device='cuda')  # encoder ids
  y_in = torch.randint(0, 1024, (2, 16), device='cuda')  # decoder inputs
  y = torch.randint(0, 1024, (2, 16), device='cuda')  # labels
  torch.cuda.reset_peak_memory_stats()  # reset stats
  out_one = model_one(x, y_in, labels=y)  # forward pass
  out_one['loss'].backward()  # backward pass
  mem_one = torch.cuda.max_memory_allocated()  # capture memory

  cfg.model.hrm.approx_grad = 'bptt'  # switch mode
  model_full = make_model(cfg).cuda()  # build bptt model
  torch.cuda.reset_peak_memory_stats()  # reset stats
  out_full = model_full(x, y_in, labels=y)  # forward pass
  out_full['loss'].backward()  # backward pass
  mem_full = torch.cuda.max_memory_allocated()  # capture memory

  assert mem_one < mem_full  # one-step should use less memory
