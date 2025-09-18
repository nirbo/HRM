#!/usr/bin/env python
"""Convert parquet dumps into HRM-LM friendly token triplets with a Hugging Face tokenizer."""

import argparse
import json
import random
from pathlib import Path
from typing import Iterator, Optional

import pyarrow.parquet as pq
from rich.progress import (
  Progress,
  SpinnerColumn,
  BarColumn,
  TimeElapsedColumn,
  TimeRemainingColumn,
  TaskProgressColumn,
  TextColumn,
)
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def clean_text(text: Optional[str]) -> Optional[str]:
  if not text:
    return None
  normalized = ' '.join(text.strip().split())
  return normalized if normalized else None


def iter_text(path: Path) -> Iterator[str]:
  pf = pq.ParquetFile(path)
  for batch in pf.iter_batches(batch_size=512, columns=['revised_text', 'text', 'language']):
    cols = batch.to_pydict()
    revised = cols.get('revised_text') or []
    original = cols.get('text') or []
    languages = cols.get('language') or []
    for rev, orig, lang in zip(revised, original, languages):
      if lang and lang != 'en':
        continue
      text = clean_text(rev) or clean_text(orig)
      if text:
        yield text


# ---------------------------------------------------------------------------
# Conversion Logic
# ---------------------------------------------------------------------------


def convert_dataset(source_dir: Path, dest_dir: Path, tokenizer_path: Path, vocab_size: int, max_seq_len: int, val_ratio: float, seed: int, max_files: Optional[int] = None) -> None:
  random.seed(seed)
  dest_dir.mkdir(parents=True, exist_ok=True)

  parquet_files = sorted(source_dir.glob('*.parquet'))
  if max_files is not None:
    parquet_files = parquet_files[:max_files]
  if not parquet_files:
    raise ValueError(f'No parquet files found in {source_dir}')

  # Train or load tokenizer -------------------------------------------------
  if tokenizer_path.exists():
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
  else:
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=['<pad>', '<bos>', '<eos>', '<unk>'])

    def text_iterator():
      for file_path in parquet_files:
        yield from iter_text(file_path)

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    tokenizer.save(str(tokenizer_path))

  pad_id = tokenizer.token_to_id('<pad>')
  bos_id = tokenizer.token_to_id('<bos>')
  eos_id = tokenizer.token_to_id('<eos>')
  if pad_id is None or bos_id is None or eos_id is None:
    raise ValueError('Tokenizer must include <pad>, <bos>, and <eos> tokens.')

  max_tokens = max(1, max_seq_len - 2)  # room for BOS/EOS
  train_path = dest_dir / 'train.jsonl'
  val_path = dest_dir / 'val.jsonl'
  meta_path = dest_dir / 'meta.json'

  train_count = 0
  val_count = 0

  progress = Progress(
    SpinnerColumn(),
    TextColumn('[progress.description]{task.description}'),
    BarColumn(),
    TaskProgressColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    transient=True,
  )

  with train_path.open('w', encoding='utf-8') as train_f, val_path.open('w', encoding='utf-8') as val_f:
    with progress:
      files_task = progress.add_task('files', total=len(parquet_files))
      samples_task = progress.add_task('samples', total=None)

      for parquet_path in parquet_files:
        for text in iter_text(parquet_path):
          tokens = tokenizer.encode(text).ids
          if not tokens:
            continue

          start = 0
          while start < len(tokens):
            chunk_tokens = tokens[start:start + max_tokens]
            start += max_tokens
            if not chunk_tokens:
              break

            seq = [bos_id] + chunk_tokens + [eos_id]
            decoder_in = seq[:-1]
            labels = seq[1:]

            sample = {
              'encoder_ids': decoder_in,
              'decoder_input_ids': decoder_in,
              'labels': labels,
            }
            target_f = train_f if random.random() > val_ratio else val_f
            json.dump(sample, target_f)
            target_f.write('\n')

            if target_f is train_f:
              train_count += 1
            else:
              val_count += 1
            progress.update(samples_task, advance=1)

        progress.update(files_task, advance=1)

  meta = {
    'max_seq_len': max_seq_len,
    'train_samples': train_count,
    'val_samples': val_count,
    'vocab_size': tokenizer.get_vocab_size(),
    'source_files': len(parquet_files),
    'tokenizer_file': str(tokenizer_path.resolve()),
    'pad_id': pad_id,
    'bos_id': bos_id,
    'eos_id': eos_id,
  }
  with meta_path.open('w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)
  print(f"Wrote {train_count} train and {val_count} val samples to {dest_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Convert parquet dataset to HRM-LM jsonl format.')
  parser.add_argument('--source', type=Path, default=Path('datasets/anothy1-fineweb-edu-cleaned-simplified'))
  parser.add_argument('--dest', type=Path, default=Path('datasets/anothy1-fineweb-edu-cleaned-simplified/processed'))
  parser.add_argument('--tokenizer', type=Path, default=None, help='Path to existing tokenizer.json; if missing, a new one is trained here.')
  parser.add_argument('--vocab-size', type=int, default=128000, help='Vocabulary size when training a new tokenizer.')
  parser.add_argument('--max-seq-len', type=int, default=256)
  parser.add_argument('--val-ratio', type=float, default=0.02)
  parser.add_argument('--seed', type=int, default=1337)
  parser.add_argument('--max-files', type=int, default=None, help='Optional limit on number of parquet files to process (useful for smoke tests).')
  args = parser.parse_args()

  tokenizer_path = args.tokenizer or (args.dest / 'tokenizer.json')
  convert_dataset(
    source_dir=args.source,
    dest_dir=args.dest,
    tokenizer_path=tokenizer_path,
    vocab_size=args.vocab_size,
    max_seq_len=args.max_seq_len,
    val_ratio=args.val_ratio,
    seed=args.seed,
    max_files=args.max_files,
  )
