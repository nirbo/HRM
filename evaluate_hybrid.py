"""Minimal evaluation entry point for the hybrid model."""

from __future__ import annotations

import argparse
import torch

from models.adapters import AdapterConfig
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1Config
from models.hybrid_hrm_transformer import HybridConfig, HybridHRMTransformer
from models.transformer_frontend import TransformerFrontEndConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid HRM+Transformer quick evaluation")
    parser.add_argument("--device", type=str, default=None, help="Device like 'cuda:0' or 'cpu'")
    parser.add_argument("--hrm-checkpoint", type=str, default=None, help="Optional path to HRM checkpoint")
    parser.add_argument("--prompt", type=str, default=None, help="Override demo prompt containing [REASON] block")
    args = parser.parse_args()

    # Resolve device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hrm_cfg = HierarchicalReasoningModel_ACTV1Config(
            batch_size=1,
            seq_len=81,
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
        adapter=AdapterConfig(transformer_dim=None, hrm_dim=None),
    )
    model = HybridHRMTransformer(config).to(device)

    if args.hrm_checkpoint:
        try:
            model.load_hrm_checkpoint(args.hrm_checkpoint)
            print(f"Loaded HRM checkpoint: {args.hrm_checkpoint}")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to load HRM checkpoint: {exc}")
    # Calc stub
    if args.prompt:
        print(model.generate(args.prompt))
        return

    # Default demos
    demo_calc = 'Answer this succinctly: [REASON] {"task":"calc","expression":"2+2*(3-1)"} [ENDREASON]'
    print(model.generate(demo_calc))

    # Sudoku routing (expects an HRM checkpoint loaded separately)
    demo_sudoku = 'Solve: [REASON] {"task":"sudoku","grid":"530070000600195000098000060800060003400803001700020006060000280000419005000080079"} [ENDREASON]'
    print(model.generate(demo_sudoku))


if __name__ == "__main__":  # pragma: no cover
    main()
