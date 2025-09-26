"""RWKV-6 encoder stack with optional CUDA acceleration.

We default to the fused CUDA kernels published in BlinkDL's RWKV-CUDA repository
(licensed under MIT, vendored in ``hrm_lm.models.rwkv_cuda``). When the kernels
or CUDA are unavailable we transparently fall back to a lightweight PyTorch
implementation so the code still works on CPU-only machines.
"""

from __future__ import annotations

import math
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
  from torch.utils.cpp_extension import load as _load_cpp_extension
except ImportError:  # pragma: no cover - torch always provides this in CI
  _load_cpp_extension = None


_CUDA_SOURCES_DIR = Path(__file__).resolve().parent / "rwkv_cuda"
_DEFAULT_VARIANT = "v1a"


def _has_cuda() -> bool:
  return torch.cuda.is_available() and _load_cpp_extension is not None


@lru_cache(maxsize=None)
def _load_wkv6_extension(head_size: int, ctx_len: int, variant: str = _DEFAULT_VARIANT):
  if not _has_cuda():
    raise RuntimeError("CUDA extension requested but CUDA is unavailable")

  op_path = _CUDA_SOURCES_DIR / "wkv6_op.cpp"
  cu_path = _CUDA_SOURCES_DIR / f"wkv6_cuda_{variant}.cu"
  if not op_path.exists() or not cu_path.exists():
    raise FileNotFoundError(
      f"Missing RWKV CUDA sources ({op_path} / {cu_path}). Ensure vendor files are present."
    )

  module_name = f"wkv6_h{head_size}_t{ctx_len}_{variant}"
  extra_cuda_cflags = [
    "-res-usage",
    "--use_fast_math",
    "-O3",
    "-Xptxas",
    "-O3",
    "--extra-device-vectorization",
    f"-D_N_={head_size}",
    f"-D_T_={ctx_len}",
  ]

  extension = _load_cpp_extension(
    name=module_name,
    sources=[str(op_path), str(cu_path)],
    extra_cflags=["-O3"],
    extra_cuda_cflags=extra_cuda_cflags,
    verbose=False,
  )
  return torch.ops.wkv6


class _WKV6Function(torch.autograd.Function):
  """Autograd wrapper around the fused CUDA kernels."""

  @staticmethod
  def forward(ctx, ops, r, k, v, w, u):
    B, T, C = r.shape
    H = u.shape[0]
    y = torch.empty_like(r)
    ops.forward(B, T, C, H, r, k, v, w, u, y)
    ctx.ops = ops
    ctx.save_for_backward(r, k, v, w, u)
    ctx.shape = (B, T, C, H)
    return y

  @staticmethod
  def backward(ctx, gy):
    ops = ctx.ops
    r, k, v, w, u = ctx.saved_tensors
    B, T, C, H = ctx.shape
    gy = gy.contiguous()

    gr = torch.empty_like(r)
    gk = torch.empty_like(k)
    gv = torch.empty_like(v)
    gw = torch.empty_like(w)
    gu_tmp = torch.empty(B, C, device=u.device, dtype=u.dtype)

    ops.backward(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu_tmp)
    gu = torch.sum(gu_tmp, dim=0).view(H, C // H)

    return (None, gr, gk, gv, gw.to(torch.float32), gu)


def _run_wkv6_cuda(ops, r, k, v, w, u):
  r_bf = r.to(torch.bfloat16).contiguous()
  k_bf = k.to(torch.bfloat16).contiguous()
  v_bf = v.to(torch.bfloat16).contiguous()
  w_fp = w.to(torch.float32).contiguous()
  u_bf = u.to(torch.bfloat16).contiguous()
  y = _WKV6Function.apply(ops, r_bf, k_bf, v_bf, w_fp, u_bf)
  return y.to(r.dtype)


class _ShiftRight(nn.Module):
  """Utility to prepend a zero token along the time dimension."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros(x.size(0), 1, x.size(2), device=x.device, dtype=x.dtype)
    return torch.cat([zeros, x[:, :-1, :]], dim=1)


class _RWKV6BlockFast(nn.Module):
  def __init__(
    self,
    d_model: int,
    n_heads: int,
    max_seq_len: int,
    ops,
    dropout: float = 0.0,
  ) -> None:
    super().__init__()
    if d_model % n_heads != 0:
      raise ValueError("d_model must be divisible by n_heads for RWKV-6")

    self.d_model = d_model
    self.n_heads = n_heads
    self.head_size = d_model // n_heads
    self.ops = ops

    self.ln = nn.LayerNorm(d_model)
    self.shift = _ShiftRight()

    self.time_mix_k = nn.Parameter(torch.rand(1, 1, d_model))
    self.time_mix_v = nn.Parameter(torch.rand(1, 1, d_model))
    self.time_mix_r = nn.Parameter(torch.rand(1, 1, d_model))
    self.time_mix_w = nn.Parameter(torch.rand(1, 1, d_model))

    self.r_proj = nn.Linear(d_model, d_model, bias=False)
    self.k_proj = nn.Linear(d_model, d_model, bias=False)
    self.v_proj = nn.Linear(d_model, d_model, bias=False)
    self.w_proj = nn.Linear(d_model, d_model, bias=False)
    self.g_proj = nn.Linear(d_model, d_model, bias=False)
    self.out_proj = nn.Linear(d_model, d_model, bias=False)

    self.time_decay = nn.Parameter(torch.full((1, 1, d_model), -3.0))
    self.time_first = nn.Parameter(torch.zeros(n_heads, self.head_size))

    self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

  def _time_mix(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_prev = self.shift(x)
    mix_k = torch.sigmoid(self.time_mix_k)
    mix_v = torch.sigmoid(self.time_mix_v)
    mix_r = torch.sigmoid(self.time_mix_r)
    mix_w = torch.sigmoid(self.time_mix_w)
    xk = x * mix_k + x_prev * (1.0 - mix_k)
    xv = x * mix_v + x_prev * (1.0 - mix_v)
    xr = x * mix_r + x_prev * (1.0 - mix_r)
    xw = x * mix_w + x_prev * (1.0 - mix_w)
    return xk, xv, xr, xw

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x_norm = self.ln(x)
    xk, xv, xr, xw = self._time_mix(x_norm)

    r = torch.sigmoid(self.r_proj(xr))
    k = self.k_proj(xk)
    v = self.v_proj(xv)
    w = (self.time_decay + self.w_proj(xw)).clamp(min=-10.0, max=10.0)
    g = torch.sigmoid(self.g_proj(xr))

    y = _run_wkv6_cuda(self.ops, r, k, v, w, self.time_first)
    y = self.dropout(self.out_proj(g * y))
    return y


class _RWKV6BlockSimple(nn.Module):
  """Portable fallback without CUDA acceleration."""

  def __init__(self, d_model: int, dropout: float = 0.0) -> None:
    super().__init__()
    self.ln = nn.LayerNorm(d_model)
    self.time_mix_k = nn.Parameter(torch.rand(d_model))
    self.time_mix_v = nn.Parameter(torch.rand(d_model))
    self.time_mix_r = nn.Parameter(torch.rand(d_model))
    self.time_decay = nn.Parameter(torch.zeros(d_model))
    self.time_first = nn.Parameter(torch.zeros(d_model))
    self.key = nn.Linear(d_model, d_model, bias=False)
    self.value = nn.Linear(d_model, d_model, bias=False)
    self.receptance = nn.Linear(d_model, d_model, bias=False)
    self.output = nn.Linear(d_model, d_model, bias=False)
    self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    xt = self.ln(x)
    B, T, C = xt.shape
    device = xt.device
    dtype = xt.dtype

    state = torch.zeros(B, C, device=device, dtype=dtype)
    prev = torch.zeros(B, C, device=device, dtype=dtype)
    outputs = []

    mix_k = self.time_mix_k.view(1, -1)
    mix_v = self.time_mix_v.view(1, -1)
    mix_r = self.time_mix_r.view(1, -1)
    decay = torch.sigmoid(self.time_decay).view(1, -1)
    first = torch.exp(self.time_first).view(1, -1)

    for t in range(T):
      x_t = xt[:, t]
      x_prev = prev
      k_t = self.key(x_t * mix_k + x_prev * (1.0 - mix_k))
      v_t = self.value(x_t * mix_v + x_prev * (1.0 - mix_v))
      r_t = torch.sigmoid(self.receptance(x_t * mix_r + x_prev * (1.0 - mix_r)))

      state = state * decay + k_t
      rwkv = (state * first + k_t) * v_t
      out_t = self.output(self.dropout(r_t * rwkv))
      outputs.append(out_t)
      prev = x_t

    y = torch.stack(outputs, dim=1)
    return y


class RWKV6Stack(nn.Module):
  def __init__(
    self,
    d_model: int,
    n_layers: int,
    n_heads: int,
    max_seq_len: int,
    dropout: float = 0.0,
    cuda_variant: str = _DEFAULT_VARIANT,
  ) -> None:
    super().__init__()
    self.d_model = d_model
    self.n_layers = n_layers
    self.n_heads = n_heads
    self.max_seq_len = max_seq_len

    self.fast_backend = False
    self.blocks = nn.ModuleList()

    if _has_cuda():
      try:
        ops = _load_wkv6_extension(d_model // n_heads, max_seq_len, cuda_variant)
        for _ in range(n_layers):
          self.blocks.append(_RWKV6BlockFast(d_model, n_heads, max_seq_len, ops, dropout))
        self.fast_backend = True
      except Exception as exc:  # pragma: no cover - CUDA fallback
        warnings.warn(f"RWKV CUDA backend unavailable ({exc}); falling back to PyTorch implementation.")

    if not self.fast_backend:
      for _ in range(n_layers):
        self.blocks.append(_RWKV6BlockSimple(d_model, dropout))

    self.norm = nn.LayerNorm(d_model)

  def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    mask = None
    if key_padding_mask is not None:
      mask = (~key_padding_mask).unsqueeze(-1).to(dtype=x.dtype)

    h = x
    for block in self.blocks:
      h = h + block(h)

    h = self.norm(h)
    if mask is not None:
      h = h * mask
    return h
