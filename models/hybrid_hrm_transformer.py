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
            # Instantiate on the transformer's device to ensure non-persistent buffers
            # (e.g., RoPE caches) are created on the correct device.
            t_device = getattr(getattr(self.transformer, "model", None), "device", torch.device("cpu"))
            with torch.device(str(t_device)):
                self.hrm = HierarchicalReasoningModel_ACTV1(hrm_cfg)  # type: ignore

        # Auto-infer adapter dims if not provided.
        t_dim = self._infer_transformer_hidden_dim()
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

    def _infer_transformer_hidden_dim(self) -> Optional[int]:
        m = getattr(self.transformer, "model", None)
        if m is None:
            return None
        cfg = getattr(m, "config", None)
        for attr in ("hidden_size", "d_model", "n_embd"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return val
        # Try embedded dim from input embeddings
        try:
            emb = m.get_input_embeddings()
            dim = getattr(emb, "embedding_dim", None)
            if isinstance(dim, int) and dim > 0:
                return dim
        except Exception:
            pass
        # Try lm_head weight shape
        lm_head = getattr(m, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight"):
            try:
                return int(lm_head.weight.shape[1])
            except Exception:
                pass
        return None

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

        Parses JSON inside the [REASON] block and dispatches either to a tiny
        stub handler (e.g., calc) or to HRM-backed domains (e.g., sudoku).
        """
        from utils.reasoning import parse_reason_content, handle_stub_task
        text = content.strip()
        try:
            obj = parse_reason_content(text)
        except Exception as exc:  # noqa: BLE001
            return f"<HRM-PARSE-ERROR:{exc}>"

        task = str(obj.get("task", "")).lower()
        if task == "sudoku":
            try:
                return self._solve_sudoku_with_hrm(obj)
            except Exception as exc:  # noqa: BLE001
                return f"<HRM-SUDOKU-ERROR:{exc}>"
        elif task == "text":
            try:
                return self._solve_text_with_hrm(obj)
            except Exception as exc:  # noqa: BLE001
                return f"<HRM-TEXT-ERROR:{exc}>"
        else:
            try:
                return handle_stub_task(obj)
            except Exception as exc:  # noqa: BLE001
                return f"<HRM-EXEC-ERROR:{exc}>"

    def _solve_sudoku_with_hrm(self, obj: dict) -> str:
        """Solve a Sudoku using the HRM model.

        JSON schema (minimal):
        {"task":"sudoku", "grid": [[..9 ints..] x9] }
        Digits 0..9 where 0 is blank.
        Returns an 81-char string of digits 0..9.
        """
        import torch

        grid = obj.get("grid")
        if grid is None:
            raise ValueError("missing 'grid'")

        # Normalize to a flat list of 81 ints
        flat: list[int] = []
        if isinstance(grid, str) and len(grid) == 81:
            for ch in grid:
                if ch == '.':
                    flat.append(0)
                elif ch.isdigit():
                    flat.append(int(ch))
                else:
                    raise ValueError(f"bad char {ch!r}")
        elif isinstance(grid, list) and len(grid) == 9 and all(isinstance(r, list) and len(r) == 9 for r in grid):
            for r in grid:
                for v in r:
                    if not isinstance(v, int) or v < 0 or v > 9:
                        raise ValueError("grid values must be ints in [0,9]")
                    flat.append(v)
        else:
            raise ValueError("grid must be a 9x9 list or an 81-char string")

        # Map to HRM tokens: 0..9 -> 1..10 (PAD=0)
        tokens = [v + 1 for v in flat]

        device = next(self.hrm.parameters()).device
        inputs = torch.tensor(tokens, dtype=torch.int32, device=device).view(1, -1)
        puzzle_identifiers = torch.zeros((1,), dtype=torch.int32, device=device)

        batch = {
            "inputs": inputs,
            "puzzle_identifiers": puzzle_identifiers,
        }

        carry = self.hrm.initial_carry(batch)
        max_steps = getattr(getattr(self.hrm, "config", None), "halt_max_steps", 1) or 1
        with torch.inference_mode():
            for _ in range(int(max_steps)):
                carry, outputs = self.hrm(carry=carry, batch=batch)
                if carry.halted.all():
                    break

        logits = outputs["logits"]  # [B, seq, vocab]
        pred = torch.argmax(logits, dim=-1)[0].to(torch.int32)  # [seq]
        # Map back to digits 0..9
        digits = (pred - 1).clamp(min=0, max=9).tolist()
        return "".join(str(int(d)) for d in digits)

    # Convenience for loading HRM weights
    def load_hrm_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=next(self.hrm.parameters()).device)
        self.hrm.load_state_dict(state, strict=False)

    def _solve_text_with_hrm(self, obj: dict) -> str:
        import torch
        from utils.text_codec import encode_bytes, decode_bytes

        prompt = obj.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("missing or invalid 'prompt'")

        # Encode prompt as byte-level tokens 1..256
        ids = encode_bytes(prompt)

        # Require HRM trained with byte vocab (PAD=0, bytes=1..256)
        vocab = getattr(getattr(self.hrm, "config", None), "vocab_size", None)
        if not isinstance(vocab, int) or vocab < 257:
            raise ValueError(f"HRM vocab_size={vocab} incompatible; expected >= 257 (byte-level)")

        device = next(self.hrm.parameters()).device
        seq_len = getattr(self.hrm.config, "seq_len", None)
        if not isinstance(seq_len, int):
            raise ValueError("HRM missing seq_len config")

        # Truncate or pad prompt to seq_len
        ids = ids[:seq_len]
        pad = [0] * (seq_len - len(ids))
        inputs = torch.tensor([ids + pad], dtype=torch.int32, device=device)
        puzzle_identifiers = torch.zeros((1,), dtype=torch.int32, device=device)

        batch = {"inputs": inputs, "puzzle_identifiers": puzzle_identifiers}
        carry = self.hrm.initial_carry(batch)

        with torch.inference_mode():
            max_steps = int(getattr(self.hrm.config, "halt_max_steps", 1) or 1)
            for _ in range(max_steps):
                carry, outputs = self.hrm(carry=carry, batch=batch)
                if carry.halted.all():
                    break

        logits = outputs["logits"]
        pred = torch.argmax(logits, dim=-1)[0].tolist()
        return decode_bytes(pred)
