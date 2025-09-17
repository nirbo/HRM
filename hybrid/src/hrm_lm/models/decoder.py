# Transformer decoder with prefix conditioning
import torch
import torch.nn as nn
from .transformer_layers import TransformerDecoder, PositionalEncoding  # include positional encoding

class LMDecoder(nn.Module):
  def __init__(self, vocab_size, d_model, n_layers, max_seq_len):
    super().__init__()
    self.tok = nn.Embedding(vocab_size, d_model)
    self.pos = PositionalEncoding(d_model, max_len=max_seq_len)  # sinusoidal positional encoding
    self.dec = TransformerDecoder(d_model, n_heads=8, n_layers=n_layers, dropout=0.0)
    self.norm = nn.LayerNorm(d_model)
    self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

  def forward(self, input_ids, memory_tokens, attention_mask=None, memory_mask=None):
    x = self.tok(input_ids)  # embed tokens
    x = self.pos(x)  # add positional encoding
    mem = memory_tokens
    h = self.dec(x, mem, tgt_key_padding_mask=(attention_mask==0) if attention_mask is not None else None, memory_key_padding_mask=memory_mask)
    h = self.norm(h)
    logits = self.lm_head(h)
    return logits
