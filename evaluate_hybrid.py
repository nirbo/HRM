"""Minimal evaluation entry point for the hybrid model."""

from __future__ import annotations

import torch

from models.adapters import AdapterConfig
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1Config
from models.hybrid_hrm_transformer import HybridConfig, HybridHRMTransformer
from models.transformer_frontend import TransformerFrontEndConfig


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hrm_cfg = HierarchicalReasoningModel_ACTV1Config(
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
        )
    config = HybridConfig(
        transformer=TransformerFrontEndConfig(device=device),
        hrm=hrm_cfg.model_dump(),
        adapter=AdapterConfig(transformer_dim=None, hrm_dim=256),
    )
    model = HybridHRMTransformer(config).to(device)
    demo = 'Answer this succinctly: [REASON] {"task":"calc","expression":"2+2*(3-1)"} [ENDREASON]'
    print(model.generate(demo))


if __name__ == "__main__":  # pragma: no cover
    main()
