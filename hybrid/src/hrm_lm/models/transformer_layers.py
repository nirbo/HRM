# minimal Transformer encoder/decoder with optional MoE feed-forward
import math
import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # TransformerEngine is optional; only required for FP8 execution
  import transformer_engine.pytorch as te  # type: ignore
except (ImportError, OSError, FileNotFoundError) as exc:  # pragma: no cover - runtime guard when TE is absent
  te = None
  warnings.warn(
    f"TransformerEngine import failed ({exc}); FP8 layers will be unavailable until the extension is installed.",
    RuntimeWarning,
    stacklevel=2,
  )


def _make_linear(
  in_features: int,
  out_features: int,
  *,
  bias: bool = False,
  use_fp8: bool = False,
  fp8_kwargs: Optional[Dict] = None,
) -> nn.Module:
  if use_fp8:
    if te is None:
      raise RuntimeError('TransformerEngine is required for FP8 linear layers but is not installed.')
    kwargs = dict(fp8_kwargs or {})
    kwargs.setdefault('params_dtype', torch.float32)
    return te.Linear(in_features, out_features, bias=bias, **kwargs)
  return nn.Linear(in_features, out_features, bias=bias)


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
    return x + self.pe[start:start + L].unsqueeze(0)


class MoEFeedForward(nn.Module):
  def __init__(
    self,
    d_model: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int = 1,
    capacity_factor: float = 1.25,
    dropout: float = 0.0,
    use_fp8: bool = False,
    fp8_linear_kwargs: Optional[Dict] = None,
  ) -> None:
    super().__init__()
    self.num_experts = num_experts
    self.top_k = max(1, top_k)
    self.capacity_factor = max(1.0, capacity_factor)
    self.hidden_dim = hidden_dim

    self.gate = nn.Linear(d_model, num_experts, bias=False)
    linear_kwargs = fp8_linear_kwargs or {}

    def build_expert():
      return nn.Sequential(
        _make_linear(d_model, hidden_dim, bias=False, use_fp8=use_fp8, fp8_kwargs=linear_kwargs),
        nn.GELU(),
        _make_linear(hidden_dim, d_model, bias=False, use_fp8=use_fp8, fp8_kwargs=linear_kwargs),
      )

    self.experts = nn.ModuleList(build_expert() for _ in range(num_experts))
    self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # x: (S, D)
    scores = torch.softmax(self.gate(x.float()), dim=-1)
    topk_vals, topk_idx = torch.topk(scores, self.top_k, dim=-1)
    S, _ = topk_idx.shape
    output = torch.zeros_like(x)

    # capacity per expert
    capacity = int(math.ceil(self.capacity_factor * (S * self.top_k) / self.num_experts))

    for expert_id, expert in enumerate(self.experts):
      selected = (topk_idx == expert_id).nonzero(as_tuple=False)  # identify tokens routed to this expert without host sync
      if selected.size(0) == 0:  # skip if graph capture finds no assignments for this expert
        continue
      token_ids = selected[:, 0]
      gate_scores = topk_vals[token_ids, selected[:, 1]]
      if capacity and token_ids.numel() > capacity:
        top_scores, order = torch.topk(gate_scores, capacity)
        token_ids = token_ids[order]
        gate_scores = top_scores
      expert_in = x[token_ids]
      expert_out = expert(expert_in)
      expert_out = self.dropout(expert_out).to(x.dtype)
      output.index_add_(0, token_ids, expert_out * gate_scores.unsqueeze(-1).to(x.dtype))

    load = torch.zeros(self.num_experts, device=x.device, dtype=x.dtype)
    for i in range(self.top_k):
      load.scatter_add_(0, topk_idx[:, i], torch.ones(S, device=x.device, dtype=x.dtype))
    importance = scores.sum(dim=0).to(x.dtype)
    aux_loss = (importance * load).sum() * self.num_experts / (S ** 2)
    return output, aux_loss


class TransformerEncoderBlock(nn.Module):
  def __init__(
    self,
    d_model: int,
    n_heads: int,
    dim_feedforward: int,
    dropout: float,
    use_moe: bool = False,
    moe_cfg: Optional[dict] = None,
    use_fp8: bool = False,
    fp8_cfg: Optional[dict] = None,
  ) -> None:
    super().__init__()
    self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    fp8_linear_kwargs = fp8_cfg.get('linear_kwargs', {}) if (use_fp8 and isinstance(fp8_cfg, dict)) else {}

    if use_moe:
      if use_fp8:
        warnings.warn('FP8 MoE experts not supported; falling back to high-precision experts.', RuntimeWarning)
      moe_cfg = moe_cfg or {}
      num_experts = int(moe_cfg.get('num_experts', 4))
      top_k = int(moe_cfg.get('top_k', 1))
      capacity_factor = float(moe_cfg.get('capacity_factor', 1.25))
      moe_dropout = float(moe_cfg.get('dropout', dropout))
      hidden_dim = int(moe_cfg.get('ff_multiplier', 4.0) * d_model)
      self.ff = MoEFeedForward(
        d_model=d_model,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
        capacity_factor=capacity_factor,
        dropout=moe_dropout,
        use_fp8=False,
        fp8_linear_kwargs=None,
      )
      self.ff_is_moe = True
    else:
      self.ff = nn.Sequential(
        _make_linear(d_model, dim_feedforward, bias=False, use_fp8=use_fp8, fp8_kwargs=fp8_linear_kwargs),
        nn.GELU(),
        nn.Dropout(dropout),
        _make_linear(dim_feedforward, d_model, bias=False, use_fp8=use_fp8, fp8_kwargs=fp8_linear_kwargs),
      )
      self.ff_is_moe = False

  def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    mask = key_padding_mask
    y = self.norm1(x)
    attn_out, _ = self.self_attn(y, y, y, key_padding_mask=mask, need_weights=False)
    x = x + self.dropout1(attn_out)

    y = self.norm2(x)
    if self.ff_is_moe:
      ff_out, aux_loss = self.ff(y.reshape(-1, y.size(-1)))
      ff_out = ff_out.reshape_as(y)
    else:
      ff_out = self.ff(y)
      aux_loss = None
    x = x + self.dropout2(ff_out)
    return x, aux_loss


class TransformerEncoder(nn.Module):
  def __init__(
    self,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dropout: float = 0.0,
    use_moe: bool = False,
    moe_cfg: Optional[dict] = None,
    use_fp8: bool = False,
    fp8_cfg: Optional[dict] = None,
  ) -> None:
    super().__init__()
    dim_feedforward = int(4 * d_model)
    self.layers = nn.ModuleList(
      TransformerEncoderBlock(
        d_model,
        n_heads,
        dim_feedforward,
        dropout,
        use_moe=use_moe,
        moe_cfg=moe_cfg,
        use_fp8=use_fp8,
        fp8_cfg=fp8_cfg,
      )
      for _ in range(n_layers)
    )
    self.norm = nn.LayerNorm(d_model)
    self.use_moe = use_moe
    self.use_fp8 = use_fp8
    self.fp8_cfg = fp8_cfg or {}

  def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    aux_loss = None
    out = x
    for layer in self.layers:
      out, layer_aux = layer(out, src_key_padding_mask)
      if layer_aux is not None:
        aux_loss = layer_aux if aux_loss is None else aux_loss + layer_aux
    out = self.norm(out)
    return out, aux_loss


class TransformerDecoder(nn.Module):
  def __init__(self, d_model, n_heads, n_layers, dropout=0.0):
    super().__init__()
    layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
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
    device = tgt.device
    dtype = tgt.dtype if tgt.dtype.is_floating_point else torch.float32
    causal = torch.zeros(L, L, device=device, dtype=dtype)
    causal.masked_fill_(torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1), float('-inf'))
    causal = causal.to(tgt.dtype) if causal.dtype != tgt.dtype else causal
    return self.dec(tgt, memory, tgt_mask=causal, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
