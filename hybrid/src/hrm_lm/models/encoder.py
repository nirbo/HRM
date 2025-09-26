# LM encoder (Transformer / Mamba2 / RWKV6)
import torch  # tensor ops
import torch.nn as nn  # neural network layers
from .transformer_layers import PositionalEncoding, TransformerEncoder  # transformer encoder building blocks
from .mamba2_layers import MambaStack  # optional Mamba2 stack
from .rwkv6_layers import RWKV6Stack  # optional RWKV-6 stack

class LMEncoder(nn.Module):
  def __init__(self, vocab_size, d_model, n_layers, max_seq_len, backend='transformer'):
    super().__init__()
    self.tok = nn.Embedding(vocab_size, d_model)
    self.pos = PositionalEncoding(d_model, max_len=max_seq_len)
    self.backend = backend
    if backend == 'mamba2':
      self.enc = MambaStack(d_model, n_layers)
    elif backend == 'rwkv6':
      self.enc = RWKV6Stack(d_model, n_layers, n_heads=8, max_seq_len=max_seq_len)
    else:
      self.enc = TransformerEncoder(d_model, n_heads=8, n_layers=n_layers, dropout=0.0)
    self.norm = nn.LayerNorm(d_model)

  def forward(self, input_ids, attention_mask=None):
    x = self.tok(input_ids)  # embed tokens
    x = self.pos(x)  # inject positional encoding
    key_pad = (attention_mask == 0) if attention_mask is not None else None  # build padding mask when provided
    h = self.enc(x, key_pad)  # encode sequence with selected backend
    h = self.norm(h)  # normalize encoder outputs for stability
    if key_pad is not None:  # honor padding mask when computing summary vector
      keep = (~key_pad).unsqueeze(-1).to(h.dtype)  # convert padding mask to multiplicative keep mask
      valid_counts = keep.sum(dim=1).clamp_min(1.0)  # count valid tokens while avoiding division by zero
      pooled = (h * keep).sum(dim=1) / valid_counts  # average valid tokens to form a stable CLS summary
      cls = pooled  # assign pooled representation
    else:
      cls = h[:, 0]  # default to first token when no padding mask exists
    return h, cls  # expose full sequence states and CLS summary
