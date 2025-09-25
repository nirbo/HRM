# thin wrapper for optional Mamba-2 encoder; falls back to Transformer
import torch
import torch.nn as nn
from .transformer_layers import TransformerEncoder


class MambaStack(nn.Module):
  def __init__(self, d_model, n_layers):
    super().__init__()
    try:
      from mamba_ssm import Mamba  # type: ignore

      self.mamba = nn.ModuleList(
        [
          Mamba(d_model=d_model, d_state=64, d_conv=4, expand=2)
          for _ in range(n_layers)
        ]
      )
      self.post_norm = nn.LayerNorm(d_model)
      self.use_mamba = True
    except Exception:
      self.enc = TransformerEncoder(
        d_model, n_heads=8, n_layers=n_layers, dropout=0.0
      )
      self.post_norm = nn.LayerNorm(d_model)
      self.use_mamba = False

  def forward(self, x, key_padding_mask=None):
    if self.use_mamba:
      original_dtype = x.dtype
      if x.dtype != torch.float32:
        x = x.to(torch.float32)

      mask = None
      if key_padding_mask is not None:
        keep = (~key_padding_mask).unsqueeze(-1).to(dtype=x.dtype)
        mask = keep
        x = x * keep

      for layer in self.mamba:
        residual = x
        updated = torch.nan_to_num(layer(x))
        if mask is not None:
          updated = updated * mask
        x = torch.nan_to_num(residual + updated)

      x = torch.nan_to_num(self.post_norm(x))
      if mask is not None:
        x = x * mask
      if x.dtype != original_dtype:
        x = x.to(original_dtype)
      return torch.nan_to_num(x)

    out = self.enc(x, src_key_padding_mask=key_padding_mask)
    out = torch.nan_to_num(out)
    return torch.nan_to_num(self.post_norm(out))
