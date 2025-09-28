#!/usr/bin/env python3
"""Download a controllable slice of a large line-oriented dataset.

Features
--------
* Works with any remote listing of dataset part files (e.g. CommonCrawl Nemotron-CC).
* Supports slicing by global sample range or approximate output size.
* Streams lines out of compressed `.jsonl.zst` parts without storing the whole dataset.
* Uses `aria2c` for multi-threaded downloads of each part file.
* Emits a companion metadata JSON describing which samples and files were included.

Example
-------
python scripts/download_dataset_slice.py \
    --index-url https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/data-jsonl.paths.gz \
    --base-url https://data.commoncrawl.org/ \
    --output nemotron_slice.jsonl \
    --start 1 --count 5_000_000 \
    --aria2c-connections 32

Dependencies
------------
* aria2c (install via `sudo apt install aria2` or `brew install aria2`).
* Python package `zstandard` (already listed in project requirements, install with pip if missing).
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import requests
import zstandard as zstd


@dataclass
class SliceRecord:
    url: str
    local_path: str
    start_sample: int
    end_sample: int
    lines_extracted: int
    skipped_lines: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a slice of a large JSONL dataset using aria2c")
    parser.add_argument("--index-url", required=True,
                        help="URL to a newline-delimited list of dataset part paths (supports .gz)")
    parser.add_argument("--base-url", default="", help="Optional base URL to prepend when entries are relative")
    parser.add_argument("--output", required=True, help="Output file (JSONL). Use .zst extension with --compress to store compressed slice")
    parser.add_argument("--metadata", default=None,
                        help="Optional path for metadata JSON. Defaults to <output>.meta.json")
    parser.add_argument("--start", type=int, default=1, help="Global 1-based start sample (default: 1)")
    range_group = parser.add_mutually_exclusive_group()
    range_group.add_argument("--end", type=int, help="Global inclusive end sample")
    range_group.add_argument("--count", type=int, help="Number of samples to extract starting from --start")
    parser.add_argument("--target-size", type=str,
                        help="Approximate maximum output size (e.g. 5G, 500M). Stops once exceeded.")
    parser.add_argument("--pattern", default=None,
                        help="Optional substring filter. Only part paths containing this text are considered.")
    parser.add_argument("--aria2c", default="aria2c", help="aria2c executable path")
    parser.add_argument("--aria2c-connections", type=int, default=16, help="Number of aria2c parallel connections (-x)")
    parser.add_argument("--aria2c-split", type=int, default=16, help="Number of aria2c download segments (-s)")
    parser.add_argument("--keep-temp", action="store_true", help="Keep downloaded part files after slicing")
    parser.add_argument("--compress", action="store_true",
                        help="Compress output slice with zstandard (.zst extension recommended)")
    parser.add_argument("--compression-level", type=int, default=3,
                        help="Zstandard compression level when --compress is set (default: 3)")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="HTTP timeout (seconds) for fetching index files")
    args = parser.parse_args()

    if args.start < 1:
        parser.error("--start must be >= 1")
    if args.end is not None and args.end < args.start:
        parser.error("--end must be >= --start")
    if args.count is not None and args.count <= 0:
        parser.error("--count must be positive")

    return args


def check_aria2c(aria2c_path: str) -> None:
    try:
        subprocess.run([aria2c_path, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"aria2c not found or unusable at '{aria2c_path}'. Install aria2c (e.g. sudo apt install aria2).") from exc


def download_index(url: str, timeout: float) -> List[str]:
    print(f"[INFO] Downloading index from {url}")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.content
    if url.endswith(".gz"):
        data = gzip.decompress(data)
    text = data.decode("utf-8")
    entries = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    print(f"[INFO] Retrieved {len(entries)} entries from index")
    return entries


def normalize_urls(entries: Iterable[str], base_url: str) -> List[str]:
    urls: List[str] = []
    for entry in entries:
        if entry.startswith("http://") or entry.startswith("https://"):
            urls.append(entry)
        else:
            if not base_url:
                raise SystemExit("Relative paths found in index; provide --base-url to construct full URLs")
            urls.append(base_url.rstrip("/") + "/" + entry.lstrip("/"))
    return urls


def parse_target_size(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    units = {"k": 1024, "m": 1024 ** 2, "g": 1024 ** 3, "t": 1024 ** 4}
    value = value.strip().lower()
    if value[-1] in units:
        return int(float(value[:-1]) * units[value[-1]])
    return int(value)


def aria2c_download(url: str, dest_dir: Path, aria2c_path: str, connections: int, split: int) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    target_path = dest_dir / filename
    if target_path.exists():
        print(f"[INFO] Reusing existing file {target_path}")
        return target_path
    existing_files = {p.name for p in dest_dir.iterdir() if p.is_file()}
    cmd = [
        aria2c_path,
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "-x", str(connections),
        "-s", str(split),
        "-k", "1M",
        "-d", str(dest_dir),
        "-o", filename,
        url,
    ]
    print(f"[INFO] Downloading {url} -> {target_path}")
    subprocess.run(cmd, check=True)
    if target_path.exists():
        return target_path

    def choose_candidate(files):
        if not files:
            return None
        return max(files, key=lambda p: (p.stat().st_size, p.stat().st_mtime))

    # Look for newly created files (aria2c may adjust the suffix)
    candidates = [
        p for p in dest_dir.iterdir()
        if p.is_file() and p.name not in existing_files and not p.name.endswith('.aria2')
    ]
    chosen = choose_candidate(candidates)
    if chosen is None:
        prefix = filename.rsplit('.', 2)[0]
        fallback = [
            p for p in dest_dir.iterdir()
            if p.is_file() and not p.name.endswith('.aria2') and p.name.startswith(prefix)
        ]
        chosen = choose_candidate(fallback)
    if chosen is not None:
        if chosen.name != filename:
            try:
                chosen.rename(target_path)
                chosen = target_path
            except OSError:
                print(f"[WARN] Unable to rename {chosen} to {target_path}; using original filename")
                return chosen
        return chosen
    # Retry with alternative extensions (e.g., .zstd -> .zst)
    alt_names = [
        filename.replace(".zstd", ".zst"),
        filename.replace(".zst", ".zstd"),
    ]
    for alt in alt_names:
        alt_path = dest_dir / alt
        if alt_path.exists():
            print(f"[WARN] Expected file {target_path} missing; using {alt_path} instead")
            return alt_path
    raise FileNotFoundError(f"aria2c completed but {target_path} not found. Check permissions or aria2c output.")


def iter_jsonl_from_zst(path: Path) -> Iterator[str]:
    suffixes = ''.join(path.suffixes).lower()
    if suffixes.endswith(".zst") or suffixes.endswith(".zstd"):
        with path.open("rb") as fh:
            dctx = zstd.ZstdDecompressor(max_window_size=2 ** 31)
            stream = dctx.stream_reader(fh)
            text_stream = io.TextIOWrapper(stream, encoding="utf-8")
            for line in text_stream:
                yield line.rstrip("\n")
    else:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                yield line.rstrip("\n")


def open_output(path: Path, compress: bool, level: int):
    if compress:
        cctx = zstd.ZstdCompressor(level=level)
        fh = path.open("wb")
        stream = cctx.stream_writer(fh)
        return fh, stream
    else:
        return None, path.open("w", encoding="utf-8")


def write_line(handle, line: str, compress: bool) -> int:
    data = (line + "\n").encode("utf-8")
    if compress:
        handle.write(data)
    else:
        handle.write(line + "\n")
    return len(data)


def main() -> None:
    args = parse_args()
    check_aria2c(args.aria2c)
    entries = download_index(args.index_url, args.timeout)
    if args.pattern:
        entries = [e for e in entries if args.pattern in e]
        print(f"[INFO] Filtered entries with pattern '{args.pattern}': {len(entries)} remaining")
    urls = normalize_urls(entries, args.base_url)

    target_bytes = parse_target_size(args.target_size)
    start = args.start
    if args.count is not None:
        target_count = args.count
    elif args.end is not None:
        target_count = args.end - args.start + 1
    else:
        target_count = None

    raw_output_path = Path(args.output).resolve()
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="dataset_slice_", dir=str(raw_output_path.parent)))
    print(f"[INFO] Using temporary directory {temp_dir}")

    if args.compress and not raw_output_path.name.endswith(".zst"):
        print("[WARN] Output extension is not .zst while --compress is set")

    base_handle, out_handle = open_output(raw_output_path, args.compress, args.compression_level)
    metadata_path = Path(args.metadata) if args.metadata else raw_output_path.with_suffix(raw_output_path.suffix + ".meta.json")

    meta_records: List[SliceRecord] = []
    current_global = 1
    written_samples = 0
    written_bytes = 0

    try:
        for url in urls:
            if target_count is not None and written_samples >= target_count:
                break
            if target_bytes is not None and written_bytes >= target_bytes:
                break

            part_path = aria2c_download(url, temp_dir, args.aria2c, args.aria2c_connections, args.aria2c_split)
            part_lines = 0
            part_written = 0
            part_skipped = 0
            part_first: Optional[int] = None
            part_last: Optional[int] = None

            for line in iter_jsonl_from_zst(part_path):
                line_global_idx = current_global
                current_global += 1
                part_lines += 1

                if line_global_idx < start:
                    part_skipped += 1
                    continue

                if target_count is not None and written_samples >= target_count:
                    break

                if target_bytes is not None and written_bytes >= target_bytes:
                    break

                bytes_added = write_line(out_handle, line, args.compress)
                written_bytes += bytes_added
                written_samples += 1
                part_written += 1
                if part_first is None:
                    part_first = line_global_idx
                part_last = line_global_idx

            if part_written:
                first = part_first if part_first is not None else 0
                last = part_last if part_last is not None else first
                meta_records.append(SliceRecord(
                    url=url,
                    local_path=str(part_path),
                    start_sample=first,
                    end_sample=last,
                    lines_extracted=part_written,
                    skipped_lines=part_skipped,
                ))
            if target_count is not None and written_samples >= target_count:
                break
            if target_bytes is not None and written_bytes >= target_bytes:
                break

            if not args.keep_temp:
                try:
                    part_path.unlink()
                except OSError:
                    pass

    finally:
        if args.compress and base_handle is not None:
            out_handle.flush()
            out_handle.close()
            base_handle.close()
        else:
            out_handle.flush()
            out_handle.close()
        if not args.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)

    metadata = {
        "index_url": args.index_url,
        "base_url": args.base_url,
        "pattern": args.pattern,
        "start": args.start,
        "count": written_samples,
        "bytes": written_bytes,
        "target_count": target_count,
        "target_bytes": target_bytes,
        "output": str(raw_output_path),
        "compress": args.compress,
        "records": [asdict(rec) for rec in meta_records],
    }

    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
        fh.write("\n")
    print(f"[INFO] Wrote {written_samples} samples ({written_bytes} bytes) to {raw_output_path}")
    print(f"[INFO] Metadata stored in {metadata_path}")


if __name__ == "__main__":
    main()
