"""
Lightweight local checks for the hybrid wrapper without external dependencies.

Runs on CPU and avoids Hugging Face and flash-attn by injecting mocks.
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.hybrid_hrm_transformer import HybridHRMTransformer, HybridConfig
from models.transformer_frontend import TransformerFrontEndConfig
from models.adapters import AdapterConfig


class _MockHRMConfig:
    hidden_size = 64


class _MockHRM:
    def __init__(self):
        self.config = _MockHRMConfig()


class _MockTok:
    def __init__(self):
        self.vocab = {"[REASON]": 1, "[ENDREASON]": 2}

    def add_special_tokens(self, *_args, **_kwargs):
        return 0

    def convert_tokens_to_ids(self, tok):
        return self.vocab.get(tok, 0)


class _MockHFModel:
    class C:
        hidden_size = 128

    config = C()
    device = "cpu"


class _MockFrontEnd:
    def __init__(self):
        self.model = _MockHFModel()
        self.tokenizer = _MockTok()
        self.special_tokens = ("[REASON]", "[ENDREASON]")

    def generate(self, prompt: str, **_):
        return f"ECHO::{prompt}"


def main():
    cfg = HybridConfig(
        transformer=TransformerFrontEndConfig(model_name="mock"),
        hrm={},
        adapter=AdapterConfig(transformer_dim=None, hrm_dim=None),
    )

    m = HybridHRMTransformer(cfg, transformer=_MockFrontEnd(), hrm_model=_MockHRM())

    # Adapter dims
    assert m.encoder.proj.in_features == 128
    assert m.encoder.proj.out_features == 64

    # Routing
    out = m.generate('Compute: [REASON] {"task":"calc","expression":"2+2*(3-1)"} [ENDREASON] now')
    assert "ECHO::" in out and "6" in out
    print("OK: routing + adapter dims")


if __name__ == "__main__":
    main()
