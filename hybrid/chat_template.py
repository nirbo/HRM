"""Universal chat template utilities for RWKV.

This module exposes helpers to:

1. Detect which major agent API (OpenAI Chat, OpenAI Responses, Anthropic Messages,
   or plain completions) a payload corresponds to.
2. Convert any of those payloads into a canonical text template using tags such as
   `<<SYS>>`, `<<USER>>`, `<<ASSISTANT>>`, `<<TOOL_CALL ...>>`, etc.
3. Reconstruct payloads for the desired API from the canonical template string.

The canonical template is identical to the one documented in ``chat_template.txt``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

__all__ = [
    "detect_api",
    "render_template",
    "template_to_payload",
]

_CANONICAL_ROLES = {
    "SYS": "system",
    "USER": "user",
    "ASSISTANT": "assistant",
    "TOOL_CALL": "tool_call",
    "TOOL_RESULT": "tool_result",
}

_START_TAG_RE = re.compile(r"^<<([A-Z_]+)([^>]*)>>$")
_END_TAG_RE = re.compile(r"^<<([A-Z_]+)_END>>$|^<<END_([A-Z_]+)>>$")
_ATTR_RE = re.compile(r"([a-zA-Z0-9_]+)=\"([^\"]*)\"")


@dataclass
class Block:
    kind: str
    text: str
    meta: Dict[str, str] = field(default_factory=dict)


def detect_api(payload) -> str:
    """Detect which API schema ``payload`` adheres to."""
    if isinstance(payload, dict):
        if "input" in payload:  # OpenAI Responses API
            return "openai-responses"
        if "messages" in payload and isinstance(payload["messages"], list):
            # Distinguish Anthropics vs OpenAI chat by message structure
            if payload["messages"] and isinstance(payload["messages"][0].get("content"), list):
                first_entry = payload["messages"][0]["content"][0]
                if isinstance(first_entry, dict) and "type" in first_entry:
                    return "anthropic"
            return "openai-chat"
        if {"prompt", "stream", "max_tokens"}.issubset(payload.keys()):
            return "openai-completions"
    if isinstance(payload, list) and payload and isinstance(payload[0], dict) and "role" in payload[0]:
        return "openai-chat"
    if isinstance(payload, str):
        return "text"
    raise ValueError("Unable to detect API schema for payload")


# ---------------------------------------------------------------------------
# Canonical serialization helpers
# ---------------------------------------------------------------------------

_TAGS = {
    "system": ("<<SYS>>", "<<SYS_END>>"),
    "user": ("<<USER>>", "<<USER_END>>"),
    "assistant": ("<<ASSISTANT>>", "<<ASSISTANT_END>>"),
    "tool_call": ("<<TOOL_CALL", "<<END_TOOL_CALL>>"),
    "tool_result": ("<<TOOL_RESULT", "<<END_TOOL_RESULT>>"),
}


def _serialize_block(block: Block) -> str:
    start, end = _TAGS[block.kind]
    if block.kind in {"tool_call", "tool_result"}:
        attrs = " ".join(f'{k}="{v}"' for k, v in block.meta.items())
        if attrs:
            start = f"{start} {attrs}>>"
        else:
            start = f"{start}>>"
    else:
        start = f"{start}"
    lines = [start, block.text.rstrip(), end]
    return "\n".join(lines)


def _parse_blocks(template: str) -> List[Block]:
    blocks: List[Block] = []
    current: Optional[Block] = None
    buffer: List[str] = []
    for raw_line in template.splitlines():
        line = raw_line.rstrip("\r")
        if not line and current is None:
            continue
        match_start = _START_TAG_RE.match(line)
        if match_start:
            if current is not None:
                raise ValueError("Nested blocks are not supported")
            tag, attr_string = match_start.groups()
            kind = _CANONICAL_ROLES.get(tag)
            if kind is None:
                raise ValueError(f"Unknown start tag {tag}")
            meta = {}
            if attr_string:
                for key, value in _ATTR_RE.findall(attr_string):
                    meta[key] = value
            current = Block(kind=kind, text="", meta=meta)
            buffer = []
            continue
        match_end = _END_TAG_RE.match(line)
        if match_end and current is not None:
            end_tag = match_end.group(1) or match_end.group(2)
            if end_tag not in _CANONICAL_ROLES:
                raise ValueError(f"Unknown end tag {line}")
            current.text = "\n".join(buffer).strip("\n")
            blocks.append(current)
            current = None
            buffer = []
            continue
        if current is None:
            continue
        buffer.append(line)
    if current is not None:
        # Allow unfinished assistant block (streaming)
        if current.kind != "assistant":
            raise ValueError("Template ended before closing block")
        current.text = "\n".join(buffer).strip("\n")
        blocks.append(current)
    return blocks


# ---------------------------------------------------------------------------
# API -> canonical blocks
# ---------------------------------------------------------------------------


def _blocks_from_openai_chat(payload: Dict) -> List[Block]:
    blocks: List[Block] = []
    for message in payload.get("messages", payload if isinstance(payload, list) else []):
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            blocks.append(Block("system", _content_to_text(content)))
        elif role == "user":
            blocks.append(Block("user", _content_to_text(content)))
        elif role == "assistant":
            func = message.get("function_call") or message.get("tool_calls")
            if func and message.get("function_call"):
                args = func.get("arguments", "{}")
                meta = {"name": func.get("name", "function")}
                if "id" in func:
                    meta["id"] = func["id"]
                blocks.append(Block("tool_call", args, meta))
            elif func and isinstance(func, list):  # tool_calls list
                for call in func:
                    meta = {"name": call.get("function", {}).get("name", "function")}
                    if call.get("id"):
                        meta["id"] = call["id"]
                    args = call.get("function", {}).get("arguments", "{}")
                    blocks.append(Block("tool_call", args, meta))
            text = _content_to_text(content)
            if text:
                blocks.append(Block("assistant", text))
        elif role == "tool":
            meta = {
                "name": message.get("name", "tool"),
            }
            if message.get("tool_call_id"):
                meta["id"] = message["tool_call_id"]
            blocks.append(Block("tool_result", _content_to_text(content), meta))
    return blocks


def _blocks_from_openai_responses(payload: Dict) -> List[Block]:
    blocks: List[Block] = []
    for entry in payload.get("input", []):
        role = entry.get("role")
        for item in entry.get("content", []):
            ctype = item.get("type")
            if ctype in {"input_text", "output_text", "text"}:
                text = item.get("text", "")
                if role == "system":
                    blocks.append(Block("system", text))
                elif role == "user":
                    blocks.append(Block("user", text))
                else:
                    blocks.append(Block("assistant", text))
            elif ctype == "tool_call":
                meta = {"name": item.get("name", "tool")}
                if item.get("id"):
                    meta["id"] = item["id"]
                blocks.append(Block("tool_call", item.get("arguments", "{}"), meta))
            elif ctype == "tool_result":
                meta = {
                    "name": item.get("name", item.get("tool_name", "tool")),
                }
                if item.get("tool_call_id"):
                    meta["id"] = item["tool_call_id"]
                if item.get("status"):
                    meta["status"] = item["status"]
                blocks.append(Block("tool_result", item.get("output", ""), meta))
    return blocks


def _blocks_from_anthropic(payload: Dict) -> List[Block]:
    blocks: List[Block] = []
    system_text = payload.get("system")
    if system_text:
        blocks.append(Block("system", system_text))
    for message in payload.get("messages", []):
        role = message.get("role")
        for item in message.get("content", []):
            ctype = item.get("type")
            if ctype == "text":
                text = item.get("text", "")
                if role == "user":
                    blocks.append(Block("user", text))
                else:
                    blocks.append(Block("assistant", text))
            elif ctype == "tool_use":
                meta = {"name": item.get("name", "tool")}
                if item.get("id"):
                    meta["id"] = item["id"]
                blocks.append(Block("tool_call", json.dumps(item.get("input", {})), meta))
            elif ctype == "tool_result":
                meta = {"name": item.get("name", "tool")}
                if item.get("tool_use_id"):
                    meta["id"] = item["tool_use_id"]
                blocks.append(Block("tool_result", _stringify(item.get("content")), meta))
    return blocks


def _blocks_from_plain_text(payload) -> List[Block]:
    template = payload if isinstance(payload, str) else str(payload)
    return [Block("user", template)]


def _content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _stringify(value) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _blocks_from_payload(payload, api: Optional[str] = None) -> List[Block]:
    api = api or detect_api(payload)
    if api == "openai-chat":
        return _blocks_from_openai_chat(payload)
    if api == "openai-responses":
        return _blocks_from_openai_responses(payload)
    if api == "anthropic":
        return _blocks_from_anthropic(payload)
    if api in {"text", "openai-completions"}:
        return _blocks_from_plain_text(payload)
    raise ValueError(f"Unsupported API: {api}")


# ---------------------------------------------------------------------------
# Canonical blocks -> API payload
# ---------------------------------------------------------------------------


def _group_blocks(blocks: List[Block]) -> List[Block]:
    """Merge consecutive system blocks into one and keep order stable."""
    merged: List[Block] = []
    for block in blocks:
        if merged and block.kind == merged[-1].kind and block.kind == "system":
            merged[-1].text += "\n\n" + block.text
        else:
            merged.append(block)
    return merged


def _build_openai_chat(blocks: List[Block]) -> Dict[str, List[Dict]]:
    messages: List[Dict] = []
    for block in blocks:
        if block.kind == "system":
            messages.append({"role": "system", "content": block.text})
        elif block.kind == "user":
            messages.append({"role": "user", "content": block.text})
        elif block.kind == "assistant":
            messages.append({"role": "assistant", "content": block.text})
        elif block.kind == "tool_call":
            msg = {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": block.meta.get("name", "function"),
                    "arguments": block.text or "{}",
                },
            }
            if block.meta.get("id"):
                msg["function_call"]["id"] = block.meta["id"]
            messages.append(msg)
        elif block.kind == "tool_result":
            msg = {
                "role": "tool",
                "name": block.meta.get("name", "tool"),
                "content": block.text,
            }
            if block.meta.get("id"):
                msg["tool_call_id"] = block.meta["id"]
            messages.append(msg)
    return {"messages": messages}


def _build_openai_responses(blocks: List[Block]) -> Dict[str, List[Dict]]:
    entries: List[Dict] = []
    for block in blocks:
        if block.kind in {"system", "user", "assistant"}:
            ctype = "input_text" if block.kind in {"system", "user"} else "output_text"
            entries.append({
                "role": block.kind,
                "content": [{"type": ctype, "text": block.text}],
            })
        elif block.kind == "tool_call":
            entry = {
                "role": "assistant",
                "content": [{
                    "type": "tool_call",
                    "name": block.meta.get("name", "tool"),
                    "arguments": block.text or "{}",
                }],
            }
            if block.meta.get("id"):
                entry["content"][0]["id"] = block.meta["id"]
            entries.append(entry)
        elif block.kind == "tool_result":
            entry = {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_call_id": block.meta.get("id"),
                    "output": block.text,
                }],
            }
            if block.meta.get("status"):
                entry["content"][0]["status"] = block.meta["status"]
            entries.append(entry)
    return {"input": entries}


def _build_anthropic(blocks: List[Block]) -> Dict[str, object]:
    system_parts: List[str] = []
    messages: List[Dict] = []
    for block in blocks:
        if block.kind == "system":
            system_parts.append(block.text)
            continue
        if block.kind in {"user", "assistant"}:
            messages.append({
                "role": block.kind,
                "content": [{"type": "text", "text": block.text}],
            })
        elif block.kind == "tool_call":
            msg = {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": block.meta.get("id", "call"),
                    "name": block.meta.get("name", "tool"),
                    "input": json.loads(block.text or "{}"),
                }],
            }
            messages.append(msg)
        elif block.kind == "tool_result":
            msg = {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": block.meta.get("id"),
                    "content": block.text,
                }],
            }
            messages.append(msg)
    result = {"messages": messages}
    if system_parts:
        result["system"] = "\n\n".join(system_parts)
    return result


def _build_plain_text(blocks: List[Block]) -> str:
    return "\n\n".join(block.text for block in blocks)


def template_to_payload(template_text: str, target_api: str) -> object:
    blocks = _parse_blocks(template_text)
    blocks = _group_blocks(blocks)
    if target_api == "openai-chat":
        return _build_openai_chat(blocks)
    if target_api == "openai-responses":
        return _build_openai_responses(blocks)
    if target_api == "anthropic":
        return _build_anthropic(blocks)
    if target_api in {"text", "openai-completions"}:
        return _build_plain_text(blocks)
    raise ValueError(f"Unsupported target API: {target_api}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_template(payload, api: Optional[str] = None) -> str:
    """Render ``payload`` into the canonical template string."""
    blocks = _blocks_from_payload(payload, api)
    blocks = _group_blocks(blocks)
    parts = [_serialize_block(block) for block in blocks]
    return "\n\n".join(parts)
