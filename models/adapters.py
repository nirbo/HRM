from __future__ import annotations

"""Simple adapter layers bridging the transformer and HRM spaces."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class AdapterConfig:
    # If transformer_dim is None/0 it will be inferred by the hybrid wrapper.
    transformer_dim: int | None
    hrm_dim: int


class EncoderAdapter(nn.Module):
    """Projects transformer hidden states into the HRM workspace."""

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(config.transformer_dim, config.hrm_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


class DecoderAdapter(nn.Module):
    """Maps HRM high-level states back to transformer dimensionality."""

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(config.hrm_dim, config.transformer_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)
