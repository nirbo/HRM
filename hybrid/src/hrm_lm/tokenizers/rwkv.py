"""RWKV tokenizer utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class _Encoding:
    """Lightweight encoding container mirroring huggingface tokenizers."""

    ids: List[int]


class _TrieNode:
    """Trie node used for greedy longest-match tokenization."""

    __slots__ = ("children", "token_id")

    def __init__(self) -> None:
        self.children: Dict[int, _TrieNode] = {}
        self.token_id: Optional[int] = None


def load_rwkv_vocab(path: str | Path) -> List[tuple[int, bytes]]:
    """Load RWKV vocabulary entries from a JSON payload."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload.get("type") not in {"rwkv_vocab_v1", "rwkv_vocab"}:
        raise ValueError("Unsupported RWKV vocabulary format")

    tokens = []
    for entry in payload.get("tokens", []):
        token_id = int(entry["id"])
        token_bytes = bytes(entry["bytes"])
        tokens.append((token_id, token_bytes))
    tokens.sort(key=lambda item: item[0])
    return tokens


class RWKVTokenizer:
    """Minimal tokenizer compatible with RWKV-7 vocabularies."""

    def __init__(self, vocab_items: Iterable[tuple[int, bytes]]) -> None:
        items = list(vocab_items)
        if not items:
            raise ValueError("RWKV vocabulary is empty")

        self._root = _TrieNode()
        self._token_to_id: Dict[bytes, int] = {}
        self._id_to_token: Dict[int, bytes] = {}
        self.pad_token_id = 0
        self.bos_token_id = 0
        self.eos_token_id = 0
        self.unk_token_id = 0

        for token_id, token_bytes in items:
            self._token_to_id[token_bytes] = token_id
            self._id_to_token[token_id] = token_bytes
            node = self._root
            for byte in token_bytes:
                node = node.children.setdefault(byte, _TrieNode())
            node.token_id = token_id

        # ensure single-byte coverage for fallback
        self._single_byte_lookup: Dict[int, int] = {}
        for token_bytes, token_id in self._token_to_id.items():
            if len(token_bytes) == 1:
                self._single_byte_lookup[token_bytes[0]] = token_id
        if len(self._single_byte_lookup) < 256:
            raise ValueError("RWKV vocabulary must contain all single-byte tokens")

    @classmethod
    def from_json(cls, path: str | Path) -> "RWKVTokenizer":
        return cls(load_rwkv_vocab(path))

    def __len__(self) -> int:
        return len(self._id_to_token) + 1  # reserve id 0 for <|endoftext|>

    @property
    def vocab_size(self) -> int:
        return len(self)

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def encode(self, text: str) -> List[int]:
        return self.encode_bytes(text.encode("utf-8"))

    def encode_bytes(self, data: bytes) -> List[int]:
        tokens: List[int] = []
        i = 0
        length = len(data)
        while i < length:
            node = self._root
            best_id: Optional[int] = None
            best_j = i
            j = i
            while j < length:
                byte = data[j]
                if byte not in node.children:
                    break
                node = node.children[byte]
                j += 1
                if node.token_id is not None:
                    best_id = node.token_id
                    best_j = j
            if best_id is None:
                byte = data[i]
                best_id = self._single_byte_lookup[byte]
                best_j = i + 1
            tokens.append(best_id)
            i = best_j
        return tokens

    def encode_batch(self, texts: Sequence[str]) -> List[_Encoding]:
        return [_Encoding(self.encode(text)) for text in texts]

    def token_to_id(self, token: str) -> Optional[int]:
        specials = {
            "<pad>": self.pad_token_id,
            "[PAD]": self.pad_token_id,
            "<bos>": self.bos_token_id,
            "<s>": self.bos_token_id,
            "<eos>": self.eos_token_id,
            "</s>": self.eos_token_id,
            "<unk>": self.unk_token_id,
            "<|endoftext|>": self.eos_token_id,
        }
        if token in specials:
            return specials[token]
        token_bytes = token.encode("utf-8")
        return self._token_to_id.get(token_bytes)

    def id_to_token(self, token_id: int) -> Optional[str]:
        if token_id == 0:
            return "<|endoftext|>"
        token_bytes = self._id_to_token.get(token_id)
        if token_bytes is None:
            return None
        try:
            return token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return None
