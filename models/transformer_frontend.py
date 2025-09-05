from __future__ import annotations

"""Transformer front-end built on top of Hugging Face models.

This module provides a thin wrapper around :class:`~transformers.AutoModelForCausalLM`
that exposes convenient encode/decode helpers and adds the special reasoning
markers used by the HRM integration (``[REASON]`` and ``[ENDREASON]``).
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn

from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TransformerFrontEndConfig:
    """Configuration for :class:`TransformerFrontEnd`.

    Attributes
    ----------
    model_name:
        Name of the Hugging Face model to load.  The default points to the
        instruction tuned Gemma 3B model, but a smaller model can be provided
        for testing.
    device:
        Device on which the model should reside.  ``None`` defaults to the
        current PyTorch default device.
    """

    model_name: str = "google/gemma-3-270m-it"
    device: Optional[torch.device] = None


class TransformerFrontEnd(nn.Module):
    """Wrapper around a pretrained causal language model.

    The front-end is responsible for handling tokenisation and providing the
    hidden states that seed the HRM reasoning module.
    """

    special_tokens: List[str] = ("[REASON]", "[ENDREASON]")

    def __init__(self, config: TransformerFrontEndConfig) -> None:
        super().__init__()

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # Add reasoning markers if they are missing.
        added = self.tokenizer.add_special_tokens({"additional_special_tokens": list(self.special_tokens)})
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name, output_hidden_states=True)
        if added:
            # Resize token embeddings to account for new special tokens.
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Ensure pad token id is set to avoid generate() warnings and inconsistent masks.
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            # Default to eos if no explicit pad exists (common for causal LMs)
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_id is not None:
                try:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    if hasattr(self.model, "config"):
                        self.model.config.pad_token_id = eos_id
                except Exception:
                    pass

        if config.device is not None:
            self.model.to(config.device)

    # ------------------------------------------------------------------
    # Tokenisation helpers
    # ------------------------------------------------------------------
    def encode(self, text: str) -> dict[str, torch.Tensor]:
        """Tokenise ``text`` and return model-ready tensors (ids + mask)."""
        return self.tokenizer(text, return_tensors="pt")

    def decode(self, ids: torch.LongTensor) -> str:
        """Decode token ids into text."""
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Model forward helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, input_ids: torch.LongTensor, **kwargs) -> torch.Tensor:
        """Run a forward pass and return the last hidden states."""
        # If caller passed only input_ids, infer attention_mask for reliability
        if "attention_mask" not in kwargs:
            kwargs["attention_mask"] = torch.ones_like(input_ids)
        outputs = self.model(input_ids=input_ids, **kwargs)
        return outputs.hidden_states[-1]

    @torch.no_grad()
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text continuation for ``prompt``."""
        toks = self.encode(prompt)
        toks = {k: v.to(self.model.device) for k, v in toks.items()}
        gen_ids = self.model.generate(**toks, **kwargs)
        return self.decode(gen_ids[0])
