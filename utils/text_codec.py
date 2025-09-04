from __future__ import annotations

"""Byte-level codec for HRM <-> text bridging.

Token map:
- 0: PAD
- 1..256: byte values 0..255
"""

from typing import Iterable, List


PAD_ID = 0


def encode_bytes(s: str) -> List[int]:
    b = s.encode("utf-8", errors="replace")
    return [x + 1 for x in b]


def decode_bytes(tokens: Iterable[int]) -> str:
    bs = bytearray()
    for t in tokens:
        if t is None:
            continue
        if t <= 0:
            continue
        bs.append((t - 1) & 0xFF)
    return bs.decode("utf-8", errors="replace")

