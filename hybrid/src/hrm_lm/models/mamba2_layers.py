# thin wrapper for optional Mamba-2 encoder; falls back to Transformer
import torch  # tensor computations
import torch.nn as nn  # neural network primitives
from .transformer_layers import TransformerEncoder  # fallback encoder for environments without Mamba

class MambaStack(nn.Module):
  def __init__(self, d_model, n_layers):
    super().__init__()  # initialize nn.Module base
    try:
      from mamba_ssm import Mamba  # type: ignore  # import the official Mamba-2 implementation when available
      self.mamba = nn.ModuleList([Mamba(d_model=d_model, d_state=64, d_conv=4, expand=2) for _ in range(n_layers)])  # construct sequential Mamba layers
      self.post_norm = nn.LayerNorm(d_model)  # align output statistics with transformer backend
      self.use_mamba = True  # flag that real Mamba kernels are active
    except Exception:
      self.enc = TransformerEncoder(d_model, n_heads=8, n_layers=n_layers, dropout=0.0)  # fall back to transformer layers on failure
      self.post_norm = nn.LayerNorm(d_model)  # still expose a norm so call sites behave uniformly
      self.use_mamba = False  # mark that fallback path is engaged

  def forward(self, x, key_padding_mask=None):
    if self.use_mamba:  # prefer Mamba execution when kernels are available
      use_fast_path = x.dtype != torch.bfloat16  # disable fused kernels when running in bf16 to avoid numerical instabilities
      mask = None  # default mask placeholder
      if key_padding_mask is not None:  # honor caller provided padding mask for stability
        keep = (~key_padding_mask).unsqueeze(-1).to(dtype=x.dtype)  # convert to multiplicative keep mask
        mask = keep  # cache the keep mask for reuse below
        x = x * keep  # zero padded tokens before entering the state space layers
      original_fast_flags = None
      if not use_fast_path:
        original_fast_flags = []
        for layer in self.mamba:
          flag = getattr(layer, 'use_fast_path', None)
          original_fast_flags.append(flag)
          if flag is not None:
            layer.use_fast_path = False  # fall back to PyTorch implementation in bf16
      for layer in self.mamba:  # iterate through each Mamba layer
        residual = x  # capture pre-layer activations for residual connection
        updated = layer(x)  # run the recurrent state space update
        if mask is not None:  # respect padding semantics immediately after each layer
          updated = updated * mask  # suppress activations originating from padded positions
        x = residual + updated  # apply residual addition to mirror transformer scaffolding
      if original_fast_flags is not None:
        for layer, flag in zip(self.mamba, original_fast_flags):
          if flag is not None:
            layer.use_fast_path = flag  # restore original fast-path configuration
      x = self.post_norm(x)  # normalize final activations for consistent scaling
      if mask is not None:  # enforce mask after normalization as well
        x = x * mask  # zero any lingering padded activations
      return x  # return masked, normalized sequence representations
    else:
      out = self.enc(x, src_key_padding_mask=key_padding_mask)  # delegate to transformer fallback with built-in masking
      return self.post_norm(out)  # normalize to match interface with Mamba path
