# HRM-LM Prototype (Mamba2/Transformer × HRM)

Minimal, end-to-end skeleton that fuses a language module (Mamba2 or Transformer) with an HRM reasoning core.
- Encoder: Mamba2 (if `mamba-ssm` installed) or Transformer encoder.
- HRM core: two-timescale recurrent module (H, L) with optional one-step gradient approx.
- Bridge: encoder→HRM and HRM→decoder (prefix tokens or cross-attn).
- Decoder: Transformer decoder.
- Training: toy loop and dry-run to validate shapes.

> Default backend runs everywhere (pure PyTorch Transformer). If `mamba-ssm` is present on Linux/CUDA, set `encoder.backend: mamba2`.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m hrm_lm.training.train --dry_run 1
```
