# Bridges between LM and HRM
import torch
import torch.nn as nn

class PromptToHRMBridge(nn.Module):
  def __init__(self, d_model):
    super().__init__()
    self.proj = nn.Linear(d_model, d_model)
  def forward(self, cls_vec):
    return self.proj(cls_vec)

class HRMToPrefixBridge(nn.Module):
  def __init__(self, d_model, prefix_len):
    super().__init__()
    self.prefix_len = prefix_len
    self.proj = nn.Linear(d_model, d_model * prefix_len)
  def forward(self, z):
    B, D = z.size()
    P = self.prefix_len
    x = self.proj(z).view(B, P, D)
    return x

class HRMToCrossAttnBridge(nn.Module):
  def __init__(self, d_model):
    super().__init__()
    self.proj = nn.Linear(d_model, d_model)
  def forward(self, z):
    return self.proj(z).unsqueeze(1)
