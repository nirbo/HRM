#!/usr/bin/env python
"""Chunk long-form text into fixed-length windows for downstream tokenization."""

import argparse
from pathlib import Path
from typing import List

from datasets import load_dataset
from transformers import AutoTokenizer


def build_argparser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description='Chunk large text corpora into fixed-length windows (token-based).')
  parser.add_argument('--input', type=Path, required=True, help='Path to a text file or glob pattern (newline-delimited).')
  parser.add_argument('--output', type=Path, required=True, help='Path to JSONL file that will hold chunked samples.')
  parser.add_argument('--tokenizer', type=str, required=True, help='Tokenizer name or path (e.g., meta-llama/Llama-3-8B-Instruct).')
  parser.add_argument('--target-length', type=int, default=1024, help='Number of tokens per chunk (default: 1024).')
  parser.add_argument('--stride', type=int, default=0, help='Token overlap between consecutive chunks (default: 0).')
  parser.add_argument('--batch-size', type=int, default=128, help='Dataset map batch size (default: 128).')
  parser.add_argument('--num-proc', type=int, default=4, help='Number of CPU processes for the map step (default: 4).')
  parser.add_argument('--truncate-short', action='store_true', help='Drop trailing windows shorter than target_length (default keeps them).')
  parser.add_argument('--keep-empty', action='store_true', help='Keep segments that collapse to empty strings after decoding (default drops them).')
  return parser


def main() -> None:
  parser = build_argparser()
  args = parser.parse_args()

  if args.stride >= args.target_length:
    raise ValueError('--stride must be smaller than --target-length to make progress.')

  tokenizer_arg = args.tokenizer
  tokenizer_path = Path(tokenizer_arg)
  if tokenizer_path.exists():
    if tokenizer_path.is_dir():
      tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    else:
      parent_dir = tokenizer_path.parent if tokenizer_path.parent.exists() else Path('.')
      tokenizer = AutoTokenizer.from_pretrained(
        str(parent_dir),
        tokenizer_file=str(tokenizer_path),
        use_fast=True,
      )
  else:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_arg, use_fast=True)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  dataset = load_dataset('text', data_files=str(args.input))['train']

  target_len = args.target_length
  stride = args.stride
  drop_short = args.truncate_short
  keep_empty = args.keep_empty

  def chunk_batch(batch):
    prompts: List[str] = []
    responses: List[str] = []
    for passage in batch['text']:
      if not passage or not passage.strip():
        continue
      tokens = tokenizer.encode(passage, add_special_tokens=False)
      if not tokens:
        continue
      start = 0
      while start < len(tokens):
        end = start + target_len
        window = tokens[start:end]
        if len(window) < target_len and drop_short:
          break
        text = tokenizer.decode(window, skip_special_tokens=True)
        if text or keep_empty:
          prompts.append(text)
          responses.append('')
        if len(window) < target_len:
          break
        start += target_len - stride if stride > 0 else target_len
    return {'prompt': prompts, 'response': responses}

  chunked = dataset.map(
    chunk_batch,
    batched=True,
    batch_size=args.batch_size,
    num_proc=args.num_proc,
    remove_columns=['text'],
  )

  total = len(chunked)
  if total == 0:
    raise RuntimeError('Chunked dataset is empty. Check tokenizer settings and overlap configuration.')

  args.output.parent.mkdir(parents=True, exist_ok=True)
  chunked.to_json(str(args.output))
  print(f'Wrote {total:,} chunks to {args.output}')


if __name__ == '__main__':
  main()
