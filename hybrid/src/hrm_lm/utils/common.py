# small helpers
import torch

def exists(x): return x is not None

def autoreg_mask(sz):
  i = torch.arange(sz)[:,None]
  j = torch.arange(sz)[None,:]
  return (i >= j).to(torch.bool)
