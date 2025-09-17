# thin wrapper for optional Mamba-2 encoder; falls back to Transformer
import torch
import torch.nn as nn
from .transformer_layers import TransformerEncoder

class MambaStack(nn.Module):
  def __init__(self, d_model, n_layers):
    super().__init__()
    try:
      from mamba_ssm import Mamba
      self.mamba = nn.ModuleList([Mamba(d_model=d_model, d_state=64, d_conv=4, expand=2) for _ in range(n_layers)])
      self.use_mamba = True
    except Exception:
      self.enc = TransformerEncoder(d_model, n_heads=8, n_layers=n_layers, dropout=0.0)
      self.use_mamba = False

  def forward(self, x, key_padding_mask=None):
    if self.use_mamba:
      for layer in self.mamba:
        x = layer(x)
      return x
    else:
      return self.enc(x, src_key_padding_mask=key_padding_mask)
