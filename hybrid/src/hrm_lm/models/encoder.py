# LM encoder (Transformer or Mamba2)
import torch  # tensor ops
import torch.nn as nn  # neural network layers
from .transformer_layers import PositionalEncoding, TransformerEncoder  # transformer encoder building blocks
from .mamba2_layers import MambaStack  # optional Mamba2 stack
from .rwkv7_backend import RWKV7Encoder  # wrap RWKV-7 modules for HRM integration

class LMEncoder(nn.Module):
  def __init__(self, vocab_size, d_model, n_layers, max_seq_len, backend='transformer', encoder_cfg=None):
    super().__init__()
    self.tok = nn.Embedding(vocab_size, d_model)
    self.pos = PositionalEncoding(d_model, max_len=max_seq_len)
    self.backend = backend
    encoder_cfg = encoder_cfg or {}
    self.supports_cuda_graphs = True  # assume CUDA graph support unless a backend opts out explicitly
    self.moe_aux_weight = float(encoder_cfg.get('moe', {}).get('aux_loss_weight', 0.0))
    fp8_cfg = encoder_cfg.pop('fp8', {}) if 'fp8' in encoder_cfg else {}
    self.fp8_enabled = bool(fp8_cfg.get('enabled', False))
    self.fp8_cfg = fp8_cfg if self.fp8_enabled else {}
    if backend == 'mamba2':
      self.enc = MambaStack(d_model, n_layers)  # Instantiate Mamba2 stack for recurrent encoder
      self.moe_aux_weight = 0.0  # Disable MoE auxiliary loss when using Mamba backend
      self.supports_cuda_graphs = False  # Mamba kernels are not CUDA-graph compatible yet
    elif backend == 'rwkv7':
      self.enc = RWKV7Encoder(vocab_size, d_model, n_layers, max_seq_len, encoder_cfg)  # Build RWKV-7 encoder wrapper using provided configuration
      self.moe_aux_weight = 0.0  # RWKV backend has no MoE auxiliary loss term
      self.fp8_enabled = False  # RWKV-7 stack manages precision internally without FP8 hooks
      self.supports_cuda_graphs = False  # RWKV-7 relies on Triton/CUDA kernels that cannot be captured safely
    else:
      moe_cfg = encoder_cfg.get('moe', {})  # Extract optional MoE configuration for transformer backend
      use_moe = bool(moe_cfg.get('enabled', False))  # Toggle MoE layer usage based on configuration flag
      self.enc = TransformerEncoder(
        d_model,
        n_heads=8,
        n_layers=n_layers,
        dropout=float(moe_cfg.get('dropout', 0.0)) if use_moe else 0.0,
        use_moe=use_moe,
        moe_cfg=moe_cfg if use_moe else None,
        use_fp8=self.fp8_enabled,
        fp8_cfg=self.fp8_cfg,
      )  # Instantiate transformer encoder with optional MoE and FP8 support
      if not use_moe:
        self.moe_aux_weight = 0.0  # Disable auxiliary loss when MoE is not active
      else:
        self.supports_cuda_graphs = False  # MoE routing uses data-dependent indexing that breaks CUDA graphs currently
    self.norm = nn.LayerNorm(d_model)

  def forward(self, input_ids, attention_mask=None):
    x = self.tok(input_ids)  # embed tokens
    x = self.pos(x)  # inject positional encoding
    key_pad = (attention_mask == 0) if attention_mask is not None else None  # build padding mask when provided
    if self.backend == 'mamba2':
      h = self.enc(x, key_pad)  # Encode sequence with Mamba stack
      aux_loss = None  # Mamba backend does not emit auxiliary losses
    elif self.backend == 'rwkv7':
      h, aux_loss = self.enc(x, key_pad)  # Encode sequence using RWKV-7 wrapper and capture placeholder aux loss
    else:
      h, aux_loss = self.enc(x, key_pad)  # Encode sequence and optional MoE auxiliary loss via transformer backend
    h = self.norm(h)  # normalize encoder outputs for stability
    if key_pad is not None:  # honor padding mask when computing summary vector
      keep = (~key_pad).unsqueeze(-1).to(h.dtype)  # convert padding mask to multiplicative keep mask
      valid_counts = keep.sum(dim=1).clamp_min(1.0)  # count valid tokens while avoiding division by zero
      pooled = (h * keep).sum(dim=1) / valid_counts  # average valid tokens to form a stable CLS summary
      cls = pooled  # assign pooled representation
    else:
      cls = h[:, 0]  # default to first token when no padding mask exists
    return h, cls, aux_loss  # expose full sequence states, CLS, and optional aux loss
