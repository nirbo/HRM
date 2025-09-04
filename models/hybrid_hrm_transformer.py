from __future__ import annotations

"""High level wrapper that combines a Transformer front-end with the HRM core."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from models.adapters import AdapterConfig, DecoderAdapter, EncoderAdapter
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1, HierarchicalReasoningModel_ACTV1Config
from models.transformer_frontend import TransformerFrontEnd, TransformerFrontEndConfig


@dataclass
class HybridConfig:
    """Configuration for the hybrid model."""

    transformer: TransformerFrontEndConfig
    hrm: HierarchicalReasoningModel_ACTV1Config
    adapter: AdapterConfig


class HybridHRMTransformer(nn.Module):
    """Wrapper orchestrating the Transformer and HRM modules.

    The implementation currently provides a minimal skeleton that forwards the
    prompt through the transformer model.  The reasoning span extraction and
    HRM execution will be implemented in subsequent iterations.
    """

    def __init__(self, config: HybridConfig) -> None:
        super().__init__()

        self.transformer = TransformerFrontEnd(config.transformer)
        self.hrm = HierarchicalReasoningModel_ACTV1(config.hrm)
        self.encoder = EncoderAdapter(config.adapter)
        self.decoder = DecoderAdapter(config.adapter)

        # Placeholder for future state caching.
        self._reason_token_id = self.transformer.tokenizer.convert_tokens_to_ids("[REASON]")
        self._end_reason_token_id = self.transformer.tokenizer.convert_tokens_to_ids("[ENDREASON]")

    @torch.no_grad()
    def generate(self, prompt: str, **gen_kwargs) -> str:
        """Generate a response.

        Currently this simply delegates to the transformer front-end.  HRM based
        reasoning for marked spans will be added later.
        """

        return self.transformer.generate(prompt, **gen_kwargs)