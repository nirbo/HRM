# LM encoder (Mamba2 or Transformer)
import torch
import torch.nn as nn
from .transformer_layers import PositionalEncoding, TransformerEncoder
from .mamba2_layers import MambaStack

class LMEncoder(nn.Module):
  def __init__(self, vocab_size, d_model, n_layers, max_seq_len, backend='transformer'):
    super().__init__()
    self.tok = nn.Embedding(vocab_size, d_model)
    self.pos = PositionalEncoding(d_model, max_len=max_seq_len)
    self.backend = backend
    if backend == 'mamba2':
      self.enc = MambaStack(d_model, n_layers)
    else:
      self.enc = TransformerEncoder(d_model, n_heads=8, n_layers=n_layers, dropout=0.0)
    self.norm = nn.LayerNorm(d_model)

  def forward(self, input_ids, attention_mask=None):
    x = self.tok(input_ids)
    x = self.pos(x)
    key_pad = (attention_mask == 0) if attention_mask is not None else None
    h = self.enc(x, key_pad)
    h = self.norm(h)
    cls = h[:, 0]
    return h, cls
