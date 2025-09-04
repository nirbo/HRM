from __future__ import annotations

"""High level wrapper that combines a Transformer front-end with the HRM core."""

from dataclasses import dataclass, replace
from typing import Optional

import torch
from torch import nn

from models.adapters import AdapterConfig, DecoderAdapter, EncoderAdapter
from models.transformer_frontend import TransformerFrontEnd, TransformerFrontEndConfig


@dataclass
class HybridConfig:
    """Configuration for the hybrid model."""

    transformer: TransformerFrontEndConfig
    # HRM config object or plain dict; instantiated lazily to avoid heavy imports during testing.
    hrm: object
    adapter: AdapterConfig


class HybridHRMTransformer(nn.Module):
    """Wrapper orchestrating the Transformer and HRM modules.

    The implementation currently provides a minimal skeleton that forwards the
    prompt through the transformer model.  The reasoning span extraction and
    HRM execution will be implemented in subsequent iterations.
    """

    def __init__(self, config: HybridConfig, *, transformer: Optional[TransformerFrontEnd] = None, hrm_model: Optional[nn.Module] = None) -> None:
        super().__init__()

        # Allow dependency injection for testing to avoid network/model downloads.
        self.transformer = transformer or TransformerFrontEnd(config.transformer)
        if hrm_model is not None:
            self.hrm = hrm_model
        else:
            # Local import to avoid mandatory flash-attn dependency during dry tests.
            from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1  # type: ignore
            hrm_cfg = config.hrm
            if hasattr(hrm_cfg, "model_dump"):
                hrm_cfg = hrm_cfg.model_dump()
            self.hrm = HierarchicalReasoningModel_ACTV1(hrm_cfg)  # type: ignore

        # Auto-infer adapter dims if not provided.
        t_hidden = getattr(getattr(self.transformer, "model", None), "config", None)
        t_dim = getattr(t_hidden, "hidden_size", None)
        h_dim = getattr(getattr(self.hrm, "config", None), "hidden_size", None)  # type: ignore

        adapter_cfg = config.adapter
        if getattr(adapter_cfg, "transformer_dim", None) in (None, 0) and t_dim is not None:
            adapter_cfg = replace(adapter_cfg, transformer_dim=t_dim)
        if getattr(adapter_cfg, "hrm_dim", None) in (None, 0) and h_dim is not None:
            adapter_cfg = replace(adapter_cfg, hrm_dim=h_dim)

        self.encoder = EncoderAdapter(adapter_cfg)
        self.decoder = DecoderAdapter(adapter_cfg)

        # Placeholder for future state caching.
        self._reason_token_id = self.transformer.tokenizer.convert_tokens_to_ids("[REASON]")
        self._end_reason_token_id = self.transformer.tokenizer.convert_tokens_to_ids("[ENDREASON]")

    @torch.no_grad()
    def generate(self, prompt: str, **gen_kwargs) -> str:
        """Generate a response.

        If the prompt contains a span delimited by [REASON] ... [ENDREASON], the
        content is routed to the HRM module and the result is substituted back
        into the prompt before delegating to the transformer front-end for fluent
        natural language phrasing.
        """
        span = self._extract_reason_span(prompt)
        if span is None:
            return self.transformer.generate(prompt, **gen_kwargs)

        prefix, content, suffix = span
        hrm_answer = self._run_hrm(content)
        routed_prompt = f"{prefix}{hrm_answer}{suffix}"
        return self.transformer.generate(routed_prompt, **gen_kwargs)

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------
    def _extract_reason_span(self, text: str) -> Optional[tuple[str, str, str]]:
        start_tok, end_tok = self.transformer.special_tokens
        start = text.find(start_tok)
        if start == -1:
            return None
        end = text.find(end_tok, start + len(start_tok))
        if end == -1:
            return None
        prefix = text[:start]
        content = text[start + len(start_tok):end]
        suffix = text[end + len(end_tok):]
        return prefix, content, suffix

    def _run_hrm(self, content: str) -> str:
        """Bridge function from natural text to HRM I/O.

        Placeholder implementation: returns a tagged echo. Will be replaced with
        domain-specific parsing and HRM execution.
        """
        return f"<HRM:{content.strip()}>"
