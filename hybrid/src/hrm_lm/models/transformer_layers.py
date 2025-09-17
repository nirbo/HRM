# minimal Transformer encoder/decoder
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=8192):
    super().__init__()
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    self.register_buffer('pe', pe, persistent=False)
  def forward(self, x, start=0):
    L = x.size(1)
    return x + self.pe[start:start+L].unsqueeze(0)

class TransformerEncoder(nn.Module):
  def __init__(self, d_model, n_heads, n_layers, dropout=0.0):
    super().__init__()
    layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
    try:
      layer._use_nested_tensor = False  # type: ignore[attr-defined]
    except AttributeError:
      pass
    try:
      layer._set_use_nested_tensor(False)  # type: ignore[attr-defined]
    except AttributeError:
      pass
    self.enc = nn.TransformerEncoder(layer, n_layers)

  def forward(self, x, src_key_padding_mask=None):
    return self.enc(x, src_key_padding_mask=src_key_padding_mask)

class TransformerDecoder(nn.Module):
  def __init__(self, d_model, n_heads, n_layers, dropout=0.0):
    super().__init__()
    layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
    try:
      layer._use_nested_tensor = False  # type: ignore[attr-defined]
    except AttributeError:
      pass
    try:
      layer._set_use_nested_tensor(False)  # type: ignore[attr-defined]
    except AttributeError:
      pass
    self.dec = nn.TransformerDecoder(layer, n_layers)

  def forward(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
    L = tgt.size(1)
    causal = torch.triu(torch.ones(L, L, device=tgt.device, dtype=torch.bool), diagonal=1)
    return self.dec(tgt, memory, tgt_mask=causal, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
