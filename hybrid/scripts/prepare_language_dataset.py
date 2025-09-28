#!/usr/bin/env python
"""Convert parquet dumps into HRM-LM friendly token triplets with a Hugging Face tokenizer."""

import argparse
import json
import random
import os
import gzip
from pathlib import Path
from typing import Iterator, List, Optional

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


def iter_parquet_text(path: Path, text_fields: Optional[List[str]]) -> Iterator[str]:
  pf = pq.ParquetFile(path)
  schema_names = set(pf.schema.names)

  candidate_fields: List[str]
  if text_fields:
    candidate_fields = [field for field in text_fields if field in schema_names]
  else:
    candidate_fields = [field for field in ['revised_text', 'text'] if field in schema_names]

  if not candidate_fields and 'raw_content' in schema_names:
    candidate_fields = ['raw_content']

  if not candidate_fields:
    return

  columns = list(candidate_fields)
  lang_col = 'language' if 'language' in schema_names else None
  if lang_col:
    columns.append(lang_col)

  for batch in pf.iter_batches(batch_size=512, columns=columns, use_threads=True):
    data = batch.to_pydict()
    first_field = candidate_fields[0]
    values_len = len(data[first_field])
    languages = data.get('language') or [None] * values_len

    for idx in range(values_len):
      lang = languages[idx]
      if lang and lang != 'en':
        continue

      text_value = None
      for field in candidate_fields:
        field_value = data[field][idx]
        if field_value:
          text_value = clean_text(field_value)
          if text_value:
            break
      if text_value:
        yield text_value



def iter_jsonl_text(path: Path) -> Iterator[str]:
  opener = gzip.open if path.suffix == '.gz' else open
  with opener(path, 'rt', encoding='utf-8') as handle:  # type: ignore[arg-type]
    for line in handle:
      line = line.strip()
      if not line:
        continue
      record = json.loads(line)
      text = build_sample_text(record)
      if text:
        yield text

def iter_json_text(path: Path) -> Iterator[str]:
  data = json.loads(path.read_text(encoding='utf-8'))
  if isinstance(data, list):
    for record in data:
      text = build_sample_text(record)
      if text:
        yield text
  else:
    text = build_sample_text(data)
    if text:
      yield text

def build_sample_text(record: dict) -> Optional[str]:
  prompt = record.get('prompt')
  response = record.get('response')
  parts = []
  if isinstance(prompt, str) and prompt.strip():
    parts.append(prompt.strip())
  if isinstance(response, str) and response.strip():
    parts.append(response.strip())
  if parts:
    return '\n'.join(parts)

  instruction = record.get('instruction')
  output = record.get('output')
  instruction_parts = []
  if isinstance(instruction, str):
    cleaned_instruction = clean_text(instruction)
    if cleaned_instruction:
      instruction_parts.append(cleaned_instruction)
  if isinstance(output, str):
    cleaned_output = clean_text(output)
    if cleaned_output:
      instruction_parts.append(cleaned_output)
  if instruction_parts:
    return '\n'.join(instruction_parts)

  conversations = record.get('conversations')
  if isinstance(conversations, list):
    conversation_parts: List[str] = []
    for turn in conversations:
      if not isinstance(turn, dict):
        continue
      value = clean_text(turn.get('value'))
      if not value:
        continue
      speaker = turn.get('from')
      if isinstance(speaker, str) and speaker.strip():
        conversation_parts.append(f"{speaker.strip().capitalize()}: {value}")
      else:
        conversation_parts.append(value)
    if conversation_parts:
      return '\n'.join(conversation_parts)

  return None

def iter_source_text(path: Path, text_fields: Optional[List[str]]) -> Iterator[str]:
  suffix = path.suffix.lower()
  if suffix == '.parquet':
    yield from iter_parquet_text(path, text_fields)
  elif suffix in {'.jsonl', '.jsonl.gz'}:
    yield from iter_jsonl_text(path)
  elif suffix == '.json':
    yield from iter_json_text(path)
  else:
    raise ValueError(f'Unsupported file format: {path}')

# ---------------------------------------------------------------------------
# Conversion Logic
# ---------------------------------------------------------------------------


def convert_dataset(source_dir: Path, dest_dir: Path, tokenizer_path: Path, vocab_size: int, tokenizer_threads: int, tokenizer_batch: int, max_seq_len: int, val_ratio: float, seed: int, max_files: Optional[int] = None, text_fields: Optional[List[str]] = None) -> None:
  random.seed(seed)
  dest_dir.mkdir(parents=True, exist_ok=True)

  patterns = ['*.parquet', '*.jsonl', '*.json', '*.jsonl.gz']
  source_files = []
  for pattern in patterns:
    source_files.extend(sorted(source_dir.glob(pattern)))
  if max_files is not None:
    source_files = source_files[:max_files]
  if not source_files:
    raise ValueError(f'No supported data files found in {source_dir}')

  if tokenizer_threads > 0:
    os.environ['RAYON_NUM_THREADS'] = str(tokenizer_threads)
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

  # Train or load tokenizer -------------------------------------------------
  if tokenizer_path.exists():
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
  else:
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(BPE(unk_token='<unk>'))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=['<pad>', '<bos>', '<eos>', '<unk>'])

    def text_iterator():
      for file_path in source_files:
        yield from iter_source_text(file_path, text_fields)

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
    TextColumn('{task.completed:,} samples', justify='right'),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    transient=True,
  )

  with train_path.open('w', encoding='utf-8') as train_f, val_path.open('w', encoding='utf-8') as val_f:
    with progress:
      files_task = progress.add_task('files', total=len(source_files))
      samples_task = progress.add_task('samples', total=None)

      for source_path in source_files:
        buffer = []
        def flush_buffer():
          nonlocal buffer, train_count, val_count
          if not buffer:
            return
          encodings = tokenizer.encode_batch(buffer)
          emitted = 0
          for encoding in encodings:
            tokens = encoding.ids
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
              emitted += 1
          progress.update(samples_task, advance=emitted)
          buffer = []

        for text in iter_source_text(source_path, text_fields):
          buffer.append(text)
          if len(buffer) >= tokenizer_batch:
            flush_buffer()

        flush_buffer()
        progress.update(files_task, advance=1)

  meta = {
    'max_seq_len': max_seq_len,
    'train_samples': train_count,
    'val_samples': val_count,
    'vocab_size': tokenizer.get_vocab_size(),
    'source_files': len(source_files),
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
  parser = argparse.ArgumentParser(description='Convert text/QA datasets (parquet/json/jsonl) into HRM-LM token triples.')
  parser.add_argument('--source', type=Path, default=Path('datasets/anothy1-fineweb-edu-cleaned-simplified'))
  parser.add_argument('--dest', type=Path, default=Path('datasets/anothy1-fineweb-edu-cleaned-simplified/processed'))
  parser.add_argument('--tokenizer', type=Path, default=None, help='Path to existing tokenizer.json; if missing, a new one is trained here.')
  parser.add_argument('--vocab-size', type=int, default=128000, help='Vocabulary size when training a new tokenizer.')
  parser.add_argument('--tokenizer-num-threads', type=int, default=0, help='Number of threads tokenizers should use (0 = library default).')
  parser.add_argument('--tokenizer-batch-size', type=int, default=256, help='Number of texts to encode per batch.')
  parser.add_argument('--max-seq-len', type=int, default=256)
  parser.add_argument('--val-ratio', type=float, default=0.02)
  parser.add_argument('--seed', type=int, default=1337)
  parser.add_argument('--max-files', type=int, default=None, help='Optional limit on number of parquet files to process (useful for smoke tests).')
  parser.add_argument('--text-field', action='append', dest='text_fields', default=None,
                      help='Name of a Parquet column to treat as text. May be provided multiple times. Defaults to revised_text/text.')
  args = parser.parse_args()

  tokenizer_path = args.tokenizer or (args.dest / 'tokenizer.json')
  convert_dataset(
    source_dir=args.source,
    dest_dir=args.dest,
    tokenizer_path=tokenizer_path,
    vocab_size=args.vocab_size,
    tokenizer_threads=args.tokenizer_num_threads,
    tokenizer_batch=args.tokenizer_batch_size,
    max_seq_len=args.max_seq_len,
    val_ratio=args.val_ratio,
    seed=args.seed,
    max_files=args.max_files,
    text_fields=args.text_fields,
  )
