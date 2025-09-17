# HRM-LM Prototype (Mamba2/Transformer × HRM)

This prototype implements a hybrid architecture that marries a lightweight language model stack with a hierarchical reasoning module (HRM). The resulting network processes natural language end-to-end while dedicating capacity to multi-step latent reasoning.

## Architecture Overview

- **Encoder**: Token embeddings plus positional encoding feed either a Transformer or Mamba2 stack (toggle via `model.encoder.backend`). The encoder produces contextual token states and a pooled `[CLS]` embedding.
- **Prompt Bridge**: A linear projection (`PromptToHRMBridge`) maps the encoder’s pooled state into the HRM latent space.
- **HRM Core**: Two coupled recurrent modules operate at different timescales: the high-level module (H) aggregates information once per cycle while the low-level module (L) iterates multiple steps guided by top-down signals. The final cycle participates in autograd (one-step gradient approximation) and emits per-cycle latents, optional halting probabilities, and the final reasoning vector.
- **Bridge to Decoder**: Depending on `bridge.type`, the HRM latent is projected into (a) prefix tokens that prepend the decoder input or (b) a single cross-attention memory vector. A learnable gate interpolates between HRM-conditioned memory and a null baseline.
- **Decoder**: A Transformer decoder consumes autoregressive inputs, conditioned on HRM memory, to produce next-token logits.
- **Training**: `hrm_lm.training.train` wires the pieces together, supports deep supervision over intermediate HRM cycles, optional halting regularization, mixed precision, validation checkpoints, and dataset iterators.

```mermaid
flowchart LR
  subgraph Encoder
    A[Input Tokens] --> B[Token Embedding]
    B --> C[Positional Encoding]
    C --> D[Transformer / Mamba2 Layers]
    D --> E[Contextual States]
    D --> F[[CLS] Pool]
  end

  F --> G[PromptToHRMBridge]
  subgraph HRM
    G --> H[High-Level Module H]
    H --> I[Top-Down Signal]
    I --> J[Low-Level Module L]
    J --> K[Bottom-Up Signal]
    K --> H
    H --> Lz[Per-Cycle Latent z_c]
  end

  Lz --> M[HRMToPrefix / CrossAttn]
  M --> N[HRM Gate]
  N --> O[Conditioned Memory]

  subgraph Decoder
    P[Decoder Input Tokens] --> Q[Embedding + Positional]
    O --> R[Transformer Decoder]
    Q --> R
    R --> S[Logits]
  end

  style HRM fill:#eef,stroke:#99f
  style Encoder fill:#efe,stroke:#6c6
  style Decoder fill:#fee,stroke:#f66
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m hrm_lm.training.train --dry_run 1
```

## Training Reference

For detailed training instructions, dataset formatting, and CLI parameter descriptions, see [`TRAINING.md`](TRAINING.md).
