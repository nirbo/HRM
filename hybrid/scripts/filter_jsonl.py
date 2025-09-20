#!/usr/bin/env python3
import argparse
import json
import multiprocessing as mp
import sys
from itertools import islice
from pathlib import Path
from typing import Iterable, List

_limit: int = 0


def _passes(sample: dict) -> bool:
    enc = sample.get("encoder_ids") or sample.get("input_ids") or []
    dec = sample.get("decoder_input_ids") or enc
    lab = sample.get("labels") or sample.get("targets") or []
    return max(len(enc), len(dec), len(lab)) <= _limit


def _init_worker(limit: int) -> None:
    global _limit
    _limit = limit


def _filter_line(line: str) -> str:
    if not line.strip():
        return ""
    sample = json.loads(line)
    if _passes(sample):
        return line if line.endswith("\n") else line + "\n"
    return ""


def _iter_chunks(handle: Iterable[str], chunk_size: int) -> Iterable[List[str]]:
    while True:
        chunk = list(islice(handle, chunk_size))
        if not chunk:
            break
        yield chunk


def filter_stream(handle: Iterable[str], dst: Path, limit: int, workers: int, chunk_size: int) -> int:
    kept = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    if workers <= 1:
        with dst.open("w", encoding="utf-8") as out:
            for line in handle:
                if not line.strip():
                    continue
                sample = json.loads(line)
                if _passes(sample):
                    if not line.endswith("\n"):
                        line += "\n"
                    out.write(line)
                    kept += 1
        return kept

    with mp.Pool(processes=workers, initializer=_init_worker, initargs=(limit,)) as pool:
        with dst.open("w", encoding="utf-8") as out:
            for chunk in _iter_chunks(handle, chunk_size):
                results = pool.map(_filter_line, chunk)
                for res in results:
                    if res:
                        out.write(res)
                        kept += 1
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=str, help='Input file or "-" for stdin')
    parser.add_argument("dst", type=Path)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=2000)
    args = parser.parse_args()

    global _limit
    _limit = args.limit

    if args.src == "-":
        handle = sys.stdin
        kept = filter_stream(handle, args.dst, args.limit, args.workers, args.chunk)
    else:
        with Path(args.src).open("r", encoding="utf-8") as handle:
            kept = filter_stream(handle, args.dst, args.limit, args.workers, args.chunk)
    print(f"kept={kept}")


if __name__ == "__main__":
    main()
