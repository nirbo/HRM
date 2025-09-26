# LM encoder (Transformer or Mamba2)
import torch  # tensor ops
import torch.nn as nn  # neural network layers
from .transformer_layers import PositionalEncoding, TransformerEncoder  # transformer encoder building blocks
from .mamba2_layers import MambaStack  # optional Mamba2 stack

class LMEncoder(nn.Module):
  def __init__(self, vocab_size, d_model, n_layers, max_seq_len, backend='transformer', encoder_cfg=None):
    super().__init__()
    self.tok = nn.Embedding(vocab_size, d_model)
    self.pos = PositionalEncoding(d_model, max_len=max_seq_len)
    self.backend = backend
    encoder_cfg = encoder_cfg or {}
    self.moe_aux_weight = float(encoder_cfg.get('moe', {}).get('aux_loss_weight', 0.0))
    if backend == 'mamba2':
      self.enc = MambaStack(d_model, n_layers)
      self.moe_aux_weight = 0.0
    else:
      moe_cfg = encoder_cfg.get('moe', {})
      use_moe = bool(moe_cfg.get('enabled', False))
      self.enc = TransformerEncoder(
        d_model,
        n_heads=8,
        n_layers=n_layers,
        dropout=float(moe_cfg.get('dropout', 0.0)) if use_moe else 0.0,
        use_moe=use_moe,
        moe_cfg=moe_cfg if use_moe else None,
      )
      if not use_moe:
        self.moe_aux_weight = 0.0
    self.norm = nn.LayerNorm(d_model)

  def forward(self, input_ids, attention_mask=None):
    x = self.tok(input_ids)  # embed tokens
    x = self.pos(x)  # inject positional encoding
    key_pad = (attention_mask == 0) if attention_mask is not None else None  # build padding mask when provided
    if self.backend == 'mamba2':
      h = self.enc(x, key_pad)  # encode sequence with selected backend
      aux_loss = None
    else:
      h, aux_loss = self.enc(x, key_pad)
    h = self.norm(h)  # normalize encoder outputs for stability
    if key_pad is not None:  # honor padding mask when computing summary vector
      keep = (~key_pad).unsqueeze(-1).to(h.dtype)  # convert padding mask to multiplicative keep mask
      valid_counts = keep.sum(dim=1).clamp_min(1.0)  # count valid tokens while avoiding division by zero
      pooled = (h * keep).sum(dim=1) / valid_counts  # average valid tokens to form a stable CLS summary
      cls = pooled  # assign pooled representation
    else:
      cls = h[:, 0]  # default to first token when no padding mask exists
    return h, cls, aux_loss  # expose full sequence states, CLS, and optional aux loss
