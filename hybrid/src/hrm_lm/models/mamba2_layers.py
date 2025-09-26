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
      self._kernel_size = 4
    except Exception:
      self.enc = TransformerEncoder(
        d_model, n_heads=8, n_layers=n_layers, dropout=0.0
      )
      self.post_norm = nn.LayerNorm(d_model)
      self.use_mamba = False
      self._kernel_size = 4

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

      orig_len = x.size(1)
      # Pad ultra-short sequences so the fused causal conv kernel (d_conv=4) never reads beyond bounds.
      pad_len = max(0, self._kernel_size - orig_len)
      if pad_len > 0:
        pad = torch.zeros(
          x.size(0), pad_len, x.size(2), dtype=x.dtype, device=x.device
        )
        x = torch.cat([x, pad], dim=1)
        if key_padding_mask is not None:
          pad_mask = torch.ones(
            key_padding_mask.size(0), pad_len, dtype=key_padding_mask.dtype, device=key_padding_mask.device
          )
          key_padding_mask = torch.cat([key_padding_mask, pad_mask.bool()], dim=1)
          keep = (~key_padding_mask).unsqueeze(-1).to(dtype=x.dtype)
          mask = keep
        else:
          key_padding_mask = torch.zeros(
            x.size(0), x.size(1), dtype=torch.bool, device=x.device
          )
          key_padding_mask[:, :orig_len] = False
          key_padding_mask[:, orig_len:] = True
          mask = (~key_padding_mask).unsqueeze(-1).to(dtype=x.dtype)
        x = x * mask

      for layer in self.mamba:
        residual = x
        updated = torch.nan_to_num(layer(x))
        if mask is not None:
          updated = updated * mask
        x = torch.nan_to_num(residual + updated)

      x = torch.nan_to_num(self.post_norm(x))
      if mask is not None:
        x = x * mask
      if pad_len > 0:
        x = x[:, :orig_len]
      if x.dtype != original_dtype:
        x = x.to(original_dtype)
      return torch.nan_to_num(x)

    out = self.enc(x, src_key_padding_mask=key_padding_mask)
    out = torch.nan_to_num(out)
    return torch.nan_to_num(self.post_norm(out))
