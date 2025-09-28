#!/usr/bin/env python3
"""Unified dataset preparation utility.

This CLI supports two main operations:
1. Extract specific fields from an input dataset into a cleaned JSONL file.
2. Optionally convert the cleaned data into token triple format
   (`encoder_ids`, `decoder_input_ids`, `labels`) for model training.

Supported input formats: JSONL (.jsonl, .jsonl.gz, .jsonl.zst/.zstd), JSON, and Parquet.

Examples
--------

Extract only the `text` field:
    python scripts/prepare_dataset.py \
        --source datasets/raw/nemotron_slice.jsonl.zst \
        --output-dir datasets/nemotron_clean \
        --fields text

Extract and tokenize into train/val/test triples:
    python scripts/prepare_dataset.py \
        --source datasets/raw/nemotron_slice.jsonl.zst \
        --output-dir datasets/nemotron_tokenized \
        --fields text \
        --to-triples \
        --text-field text \
        --tokenizer tokenizer.json \
        --max-seq-len 512 \
        --val-ratio 0.01 \
        --test-ratio 0.01
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import pyarrow.parquet as pq
import zstandard as zstd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm


# ----------------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------------

SUPPORTED_PATTERNS = [
    "*.jsonl", "*.jsonl.gz", "*.jsonl.zst", "*.jsonl.zstd",
    "*.json", "*.parquet", "*.zst", "*.zstd"
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic dataset preparation tool")
    parser.add_argument("--source", required=True,
                        help="Input file or directory containing JSONL/JSON/Parquet data")
    parser.add_argument("--output-dir", required=True,
                        help="Directory where outputs will be written")
    parser.add_argument("--fields", nargs="+", required=True,
                        help="Fields to keep in the cleaned JSONL")
    parser.add_argument("--template", default=None,
                        help="Optional Python format string to build an additional field using source keys")
    parser.add_argument("--template-field", default="text",
                        help="Name of the field populated by --template (default: text)")
    parser.add_argument("--batch-size", type=int, default=8192,
                        help="Batch size used when encoding to triples (default: 8192)")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable tqdm progress bars")
    parser.add_argument("--count-records", action="store_true",
                        help="Pre-count records to display total progress (requires a full pre-pass)")
    parser.add_argument("--force-extract", action="store_true",
                        help="Redo extraction even if extracted.jsonl already exists")
    parser.add_argument("--force-triples", action="store_true",
                        help="Redo triple conversion even if train/val/test files already exist")

    # Triple conversion
    parser.add_argument("--to-triples", action="store_true", help="Convert cleaned data into train/val/test triples")
    parser.add_argument("--text-field", default=None,
                        help="Field name whose content is tokenized when --to-triples is set")
    parser.add_argument("--tokenizer", default=None,
                        help="Path to tokenizer.json (trained or to be created)")
    parser.add_argument("--vocab-size", type=int, default=128000,
                        help="Vocabulary size when training a new tokenizer")
    parser.add_argument("--tokenizer-num-threads", type=int, default=0,
                        help="Number of threads tokenizers should use (0 = library default)")
    parser.add_argument("--tokenizer-batch-size", type=int, default=8192,
                        help="Number of texts to encode per batch when tokenizing")
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="Maximum sequence length (tokens) for triples")
    parser.add_argument("--val-ratio", type=float, default=0.01,
                        help="Validation holdout ratio when no explicit split exists")
    parser.add_argument("--test-ratio", type=float, default=0.0,
                        help="Test holdout ratio when no explicit split exists")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for holdout selection")

    return parser.parse_args()


# ----------------------------------------------------------------------------
# Reading datasets
# ----------------------------------------------------------------------------


def collect_source_files(source: Path) -> List[Path]:
    if source.is_file():
        return [source]
    files: List[Path] = []
    for pattern in SUPPORTED_PATTERNS:
        files.extend(sorted(source.glob(pattern)))
    if not files:
        raise FileNotFoundError(f"No supported files found at {source}")
    return files


def iter_records(path: Path) -> Iterator[Dict[str, object]]:
    suffixes = ''.join(path.suffixes).lower()
    if suffixes.endswith('.jsonl.zst') or suffixes.endswith('.jsonl.zstd') or suffixes.endswith('.zst') or suffixes.endswith('.zstd'):
        with path.open('rb') as fh:
            dctx = zstd.ZstdDecompressor(max_window_size=2 ** 31)
            stream = dctx.stream_reader(fh)
            text_stream = io.TextIOWrapper(stream, encoding='utf-8')
            for line in text_stream:
                if not line.strip():
                    continue
                yield json.loads(line)
    elif suffixes.endswith('.jsonl.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                yield json.loads(line)
    elif suffixes.endswith('.jsonl'):
        with path.open('r', encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                yield json.loads(line)
    elif suffixes.endswith('.json'):
        data = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
        elif isinstance(data, dict):
            yield data
    elif suffixes.endswith('.parquet'):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches():
            table = batch.to_pylist()
            for row in table:
                if isinstance(row, dict):
                    yield row
    else:
        raise ValueError(f"Unsupported file type: {path}")


# ----------------------------------------------------------------------------
# Cleaning / extraction
# ----------------------------------------------------------------------------


def build_clean_record(record: Dict[str, object], keep_fields: List[str], template: Optional[str], template_field: str) -> Optional[Dict[str, object]]:
    cleaned: Dict[str, object] = {}
    for field in keep_fields:
        value = record.get(field)
        if value is not None:
            cleaned[field] = value
    if template:
        try:
            rendered = template.format(**record)
        except KeyError:
            return None
        cleaned[template_field] = rendered
    return cleaned if cleaned else None


def count_records(files: List[Path]) -> int:
    total = 0
    for file_path in files:
        for _ in iter_records(file_path):
            total += 1
    return total


def write_clean_jsonl(source: Path, files: List[Path], keep_fields: List[str], template: Optional[str], template_field: str, output_dir: Path, show_progress: bool, count_total: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'extracted.jsonl'
    count = 0
    total = None
    if show_progress and count_total:
        print('[INFO] Counting records for extraction progress meter (full pass)...')
        total = count_records(files)
    bar_format = '{l_bar}{bar}| {n:,} records [{elapsed}, {rate_fmt}]'
    if total:
        bar_format = '{l_bar}{bar}| {n:,} / {total:,} records [{elapsed} < {remaining}, {rate_fmt}]'
    progress = tqdm(
        disable=not show_progress,
        desc='Extracting',
        unit='records',
        total=total,
        colour='green',
        bar_format=bar_format
    )
    with output_path.open('w', encoding='utf-8') as out:
        for file_path in files:
            for record in iter_records(file_path):
                cleaned = build_clean_record(record, keep_fields, template, template_field)
                if cleaned is None:
                    continue
                out.write(json.dumps(cleaned, ensure_ascii=False) + '\n')
                count += 1
                progress.update(1)
    progress.close()
    print(f"[INFO] Wrote {count:,} records to {output_path}")
    return output_path


# ----------------------------------------------------------------------------
# Tokenizer helpers
# ----------------------------------------------------------------------------


def ensure_tokenizer(args: argparse.Namespace, dataset_path: Path) -> Tokenizer:
    tokenizer_path = Path(args.tokenizer)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

    if tokenizer_path.exists():
        print(f"[INFO] Loading tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        print(f"[INFO] Training new tokenizer at {tokenizer_path}")
        tokenizer = Tokenizer(BPE(unk_token='<unk>'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(vocab_size=args.vocab_size, special_tokens=['<pad>', '<bos>', '<eos>', '<unk>'])

        def iterator():
            with dataset_path.open('r', encoding='utf-8') as handle:
                for line in handle:
                    record = json.loads(line)
                    text = record.get(args.text_field)
                    if text:
                        yield str(text)

        tokenizer.train_from_iterator(iterator(), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
        print(f"[INFO] Saved tokenizer to {tokenizer_path}")

    return tokenizer


# ----------------------------------------------------------------------------
# Triple conversion
# ----------------------------------------------------------------------------


def choose_split(rng: random.Random, val_ratio: float, test_ratio: float) -> str:
    r = rng.random()
    if r < test_ratio:
        return 'test'
    if r < test_ratio + val_ratio:
        return 'val'
    return 'train'


def convert_to_triples(clean_path: Path, output_dir: Path, args: argparse.Namespace, show_progress: bool, count_total: bool) -> None:
    if not args.tokenizer:
        raise SystemExit("--tokenizer is required when --to-triples is specified")
    if not args.text_field:
        raise SystemExit("--text-field must be provided when --to-triples is set")
    if args.val_ratio + args.test_ratio >= 1.0:
        raise SystemExit("val_ratio + test_ratio must be < 1.0")

    tokenizer = ensure_tokenizer(args, clean_path)

    if args.tokenizer_num_threads > 0:
        os.environ['RAYON_NUM_THREADS'] = str(args.tokenizer_num_threads)
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    pad_id = tokenizer.token_to_id('<pad>')
    bos_id = tokenizer.token_to_id('<bos>')
    eos_id = tokenizer.token_to_id('<eos>')
    if pad_id is None or bos_id is None or eos_id is None:
        raise SystemExit('Tokenizer must include <pad>, <bos>, and <eos> tokens')

    max_tokens = max(1, args.max_seq_len - 2)
    splits = {
        'train': (output_dir / 'train.jsonl').open('w', encoding='utf-8'),
        'val': (output_dir / 'val.jsonl').open('w', encoding='utf-8'),
        'test': (output_dir / 'test.jsonl').open('w', encoding='utf-8'),
    }
    counts = {'train': 0, 'val': 0, 'test': 0}
    rng = random.Random(args.seed)

    total = None
    if show_progress and count_total:
        print('[INFO] Counting records in cleaned file for conversion progress...')
        with clean_path.open('r', encoding='utf-8') as handle:
            total = sum(1 for _ in handle)
    bar_format = '{l_bar}{bar}| {n:,} records [{elapsed}, {rate_fmt}]'
    if total:
        bar_format = '{l_bar}{bar}| {n:,} / {total:,} records [{elapsed} < {remaining}, {rate_fmt}]'
    progress = tqdm(
        disable=not show_progress,
        desc='Converting',
        unit='records',
        total=total,
        colour='green',
        bar_format=bar_format
    )

    def encode_batch(records: List[Dict[str, object]], texts: List[str]):
        encodings = tokenizer.encode_batch(texts)
        for record, encoding in zip(records, encodings):
            ids = encoding.ids[:max_tokens]
            seq = [bos_id] + ids + [eos_id]
            triple = {
                'encoder_ids': seq[:-1],
                'decoder_input_ids': seq[:-1],
                'labels': seq[1:],
            }
            split = record['split']
            json.dump(triple, splits[split])
            splits[split].write('\n')
            counts[split] += 1
            progress.update(1)

    batch_records: List[Dict[str, object]] = []
    batch_texts: List[str] = []

    with clean_path.open('r', encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            value = record.get(args.text_field)
            if not value:
                continue
            split = choose_split(rng, args.val_ratio, args.test_ratio)
            batch_records.append({'split': split})
            batch_texts.append(str(value))
            if len(batch_texts) >= args.tokenizer_batch_size:
                encode_batch(batch_records, batch_texts)
                batch_records.clear()
                batch_texts.clear()

    if batch_texts:
        encode_batch(batch_records, batch_texts)

    progress.close()

    for fh in splits.values():
        fh.close()

    meta = {
        'max_seq_len': args.max_seq_len,
        'train_samples': counts['train'],
        'val_samples': counts['val'],
        'test_samples': counts['test'],
        'vocab_size': tokenizer.get_vocab_size(),
        'tokenizer_file': str(Path(args.tokenizer).resolve()),
        'pad_id': pad_id,
        'bos_id': bos_id,
        'eos_id': eos_id,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
    }
    with (output_dir / 'meta.json').open('w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=2)
        fh.write('\n')
    print(f"[INFO] Triple conversion complete: {counts}")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = collect_source_files(source)
    show_progress = not args.no_progress
    clean_path = output_dir / 'extracted.jsonl'
    if clean_path.exists() and not args.force_extract:
        print(f"[INFO] Found existing {clean_path}; skipping extraction (use --force-extract to redo)")
    else:
        clean_path = write_clean_jsonl(
            source,
            files,
            args.fields,
            args.template,
            args.template_field,
            output_dir,
            show_progress,
            args.count_records,
        )

    if args.to_triples:
        triple_outputs = [
            output_dir / 'train.jsonl',
            output_dir / 'val.jsonl',
            output_dir / 'test.jsonl',
            output_dir / 'meta.json',
        ]
        if all(path.exists() for path in triple_outputs) and not args.force_triples:
            print(f"[INFO] Triple outputs already exist in {output_dir}; skipping conversion (use --force-triples to redo)")
        else:
            convert_to_triples(clean_path, output_dir, args, show_progress, args.count_records)


if __name__ == '__main__':
    main()
