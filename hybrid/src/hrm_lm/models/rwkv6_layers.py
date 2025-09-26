# Lightweight RWKV-6 style recurrent stack for HRM integrations.
import torch
import torch.nn as nn


class _RWKV6Block(nn.Module):
  def __init__(self, d_model, dropout=0.0):
    super().__init__()
    self.time_mix_k = nn.Parameter(torch.rand(d_model))
    self.time_mix_v = nn.Parameter(torch.rand(d_model))
    self.time_mix_r = nn.Parameter(torch.rand(d_model))
    self.time_decay = nn.Parameter(torch.zeros(d_model))
    self.time_first = nn.Parameter(torch.zeros(d_model))
    self.key = nn.Linear(d_model, d_model, bias=False)
    self.value = nn.Linear(d_model, d_model, bias=False)
    self.receptance = nn.Linear(d_model, d_model, bias=False)
    self.output = nn.Linear(d_model, d_model, bias=False)
    self.ln = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

  def forward(self, x, mask=None):
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
      r_t = torch.sigmoid(
        self.receptance(x_t * mix_r + x_prev * (1.0 - mix_r))
      )

      state = state * decay + k_t
      rwkv = (state * first + k_t) * v_t
      out_t = self.output(self.dropout(r_t * rwkv))
      outputs.append(out_t)
      prev = x_t

    y = torch.stack(outputs, dim=1)
    if mask is not None:
      y = y * mask
    return x + y


class RWKV6Stack(nn.Module):
  def __init__(self, d_model, n_layers, dropout=0.0):
    super().__init__()
    self.layers = nn.ModuleList(
      [_RWKV6Block(d_model, dropout=dropout) for _ in range(n_layers)]
    )
    self.norm = nn.LayerNorm(d_model)

  def forward(self, x, key_padding_mask=None):
    mask = None
    if key_padding_mask is not None:
      mask = (~key_padding_mask).unsqueeze(-1).to(dtype=x.dtype)

    h = x
    for layer in self.layers:
      h = layer(h, mask=mask)

    h = self.norm(h)
    if mask is not None:
      h = h * mask
    return h
