# HRM core with H/L recurrent modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_layers import TransformerEncoder

class HRMModule(nn.Module):
  def __init__(self, d_model, mem_len, n_layers):
    super().__init__()
    self.mem_len = mem_len
    self.pos = nn.Parameter(torch.zeros(mem_len, d_model))
    self.block = TransformerEncoder(d_model, n_heads=8, n_layers=n_layers, dropout=0.0)
    self.norm = nn.LayerNorm(d_model)

  def forward(self, state, extra=None, key_padding_mask=None):
    if extra is not None:
      state = state + extra
    x = state + self.pos.unsqueeze(0)
    x, _ = self.block(x, src_key_padding_mask=key_padding_mask)
    return self.norm(x)

class HRMCore(nn.Module):
  def __init__(self, d_model=512, h_len=8, l_len=64, h_layers=2, l_layers=2, h_cycles=4, l_steps=8, approx_grad='one_step', out_dim=512, use_halting=False):
    super().__init__()
    self.d_model = d_model
    self.h_cycles = h_cycles
    self.l_steps = l_steps
    self.approx_grad = approx_grad
    self.use_halting = use_halting

    self.inp_to_h = nn.Linear(d_model, d_model)
    self.inp_to_l = nn.Linear(d_model, d_model)

    self.h_mod = HRMModule(d_model, h_len, h_layers)
    self.l_mod = HRMModule(d_model, l_len, l_layers)

    self.topdown = nn.Linear(d_model, d_model)
    self.bottomup = nn.Linear(d_model, d_model)

    self.h_init = nn.Parameter(torch.zeros(1, h_len, d_model))
    self.l_init = nn.Parameter(torch.zeros(1, l_len, d_model))

    self.out_norm = nn.LayerNorm(d_model)
    self.out = nn.Linear(d_model, out_dim)

    if use_halting:
      self.halt_head = nn.Sequential(nn.Linear(d_model, 1))

  def forward(self, x_enc, return_all=False):
    B = x_enc.size(0)
    hin = self.inp_to_h(x_enc).unsqueeze(1)
    lin = self.inp_to_l(x_enc).unsqueeze(1)

    H = self.h_init.expand(B, -1, -1)
    L = self.l_init.expand(B, -1, -1)

    halting_probs = []
    z_list = []

    for c in range(self.h_cycles):  # iterate over high-level recurrent cycles
      truncated = self.approx_grad == 'one_step' and c < self.h_cycles - 1 and not self.use_halting  # respect approx grad truncation only when halting is disabled
      if truncated:  # apply gradient truncation for early cycles when allowed
        with torch.no_grad():  # disable gradient tracking for cheaper inference-style updates
          L = self._run_L(L, H, lin)  # advance low-level memory without gradient tracking
          H = self._run_H(H, L, hin)  # advance high-level memory without gradient tracking
      else:
        L = self._run_L(L, H, lin)  # run low-level memory with gradients
        H = self._run_H(H, L, hin)  # run high-level memory with gradients

      z_raw = self._readout(H)  # compute raw latent
      z_norm = self.out_norm(z_raw)  # normalize latent
      z_proj = self.out(z_norm)  # project to decoder space
      z_list.append(z_proj)  # store projected latent
      if self.use_halting:
        p = torch.sigmoid(self.halt_head(H[:,0]))
        halting_probs.append(p)

    z_final = z_list[-1]  # pick last projected latent
    out = z_final  # final HRM output

    aux = {'z_per_cycle': z_list, 'halt': halting_probs} if return_all else None
    return out, aux

  def _run_L(self, L, H, lin):
    td = self.topdown(H.mean(dim=1)).unsqueeze(1).expand_as(L)
    extra = td + lin.expand_as(L)
    for t in range(self.l_steps):
      L = self.l_mod(L, extra=extra)
    return L

  def _run_H(self, H, L, hin):
    bu = self.bottomup(L.mean(dim=1)).unsqueeze(1).expand_as(H)
    extra = bu + hin.expand_as(H)
    H = self.h_mod(H, extra=extra)
    return H

  def _readout(self, H):
    return H.mean(dim=1)
