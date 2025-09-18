#!/usr/bin/env python
"""Convert parquet dumps into HRM-LM friendly token triplets."""

import argparse
import json
import random
from pathlib import Path
from typing import Iterator, Optional

import pyarrow.parquet as pq
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn, TextColumn

from hrm_lm.data.simple_tokenizer import SimpleTokenizer, BOS_ID, EOS_ID


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


def convert_dataset(source_dir: Path, dest_dir: Path, max_seq_len: int, val_ratio: float, seed: int) -> None:
  random.seed(seed)
  dest_dir.mkdir(parents=True, exist_ok=True)

  tokenizer = SimpleTokenizer()
  max_tokens = max_seq_len - 1
  train_path = dest_dir / 'train.jsonl'
  val_path = dest_dir / 'val.jsonl'

  train_count = 0
  val_count = 0

  with train_path.open('w', encoding='utf-8') as train_f, val_path.open('w', encoding='utf-8') as val_f:
    parquet_files = sorted(source_dir.glob('*.parquet'))
    progress = Progress(
      SpinnerColumn(),
      TextColumn('[progress.description]{task.description}'),
      BarColumn(),
      TaskProgressColumn(),
      TimeElapsedColumn(),
      TimeRemainingColumn(),
      transient=True,
    )
    with progress:
      files_task = progress.add_task('files', total=len(parquet_files))
      samples_task = progress.add_task('samples', total=None)
      for parquet_path in parquet_files:
        for text in iter_text(parquet_path):
          tokenizer.fit([text])
          tokens = tokenizer.encode(text, add_specials=False)
          if not tokens:
            continue
          start = 0
          while start < len(tokens):
            chunk_tokens = tokens[start:start + max_tokens]
            start += max_tokens
            if not chunk_tokens:
              break
            seq = [tokenizer.bos_id] + chunk_tokens + [tokenizer.eos_id]
            if len(seq) < 3:
              continue
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

  vocab_path = dest_dir / 'tokenizer.json'
  with vocab_path.open('w', encoding='utf-8') as f:
    json.dump({'stoi': tokenizer.stoi}, f)

  meta = {
    'max_seq_len': max_seq_len,
    'train_samples': train_count,
    'val_samples': val_count,
    'vocab_size': len(tokenizer.stoi),
    'source_files': len(list(source_dir.glob('*.parquet'))),
  }
  with (dest_dir / 'meta.json').open('w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)
  print(f'Wrote {train_count} train and {val_count} val samples to {dest_dir}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Convert parquet dataset to HRM-LM jsonl format.')
  parser.add_argument('--source', type=Path, default=Path('datasets/anothy1-fineweb-edu-cleaned-simplified'))
  parser.add_argument('--dest', type=Path, default=Path('datasets/anothy1-fineweb-edu-cleaned-simplified/processed'))
  parser.add_argument('--max-seq-len', type=int, default=256)
  parser.add_argument('--val-ratio', type=float, default=0.02)
  parser.add_argument('--seed', type=int, default=1337)
  args = parser.parse_args()

  convert_dataset(args.source, args.dest, max_seq_len=args.max_seq_len, val_ratio=args.val_ratio, seed=args.seed)
