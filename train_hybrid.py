"""Training script for the hybrid Transformer + HRM model.

This script provides a minimal example of how the different modules can be
initialised and optimised jointly.  The actual dataset handling and training
loops are intentionally simplified and should be expanded in future work.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from models.adapters import AdapterConfig
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1Config
from models.hybrid_hrm_transformer import HybridConfig, HybridHRMTransformer
from models.transformer_frontend import TransformerFrontEndConfig


class DummyDataset(Dataset):
    """Small dummy dataset used for demonstration purposes."""

    def __len__(self) -> int:  # pragma: no cover - trivial
        return 2

    def __getitem__(self, idx: int):  # pragma: no cover - trivial
        return "Solve 1+1", "2"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = HybridConfig(
        transformer=TransformerFrontEndConfig(device=device),
        hrm=HierarchicalReasoningModel_ACTV1Config(
            batch_size=1,
            seq_len=32,
            puzzle_emb_ndim=0,
            num_puzzle_identifiers=1,
            vocab_size=32000,
            H_cycles=1,
            L_cycles=1,
            H_layers=1,
            L_layers=1,
            hidden_size=256,
            expansion=4.0,
            num_heads=8,
            pos_encodings="rope",
            halt_max_steps=1,
            halt_exploration_prob=0.0,
        ),
        adapter=AdapterConfig(transformer_dim=256, hrm_dim=256),
    )

    model = HybridHRMTransformer(config).to(device)
    optimiser = optim.AdamW(model.parameters(), lr=1e-4)

    dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for epoch in range(1):
        for prompt, target in loader:
            optimiser.zero_grad()
            output = model.generate(prompt[0])  # Currently transformer only
            loss = torch.tensor(0.0, requires_grad=True, device=device)
            loss.backward()
            optimiser.step()
            print(f"Epoch {epoch} loss {loss.item():.4f} output={output}")


if __name__ == "__main__":  # pragma: no cover
    main()