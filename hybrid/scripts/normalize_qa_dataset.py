#!/usr/bin/env python
"""Normalize QA-style datasets into prompt/response JSONL records."""
import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pyarrow.parquet as pq

Record = Dict[str, Any]

THREAD_ENV_VARS = ('RAYON_NUM_THREADS', 'TOKENIZERS_PARALLELISM')



def _tokenize_field(expr: str) -> Sequence[Union[str, int]]:
  tokens: List[Union[str, int]] = []
  for part in expr.split('.'):
    while part:
      if '[' in part:
        prefix, suffix = part.split('[', 1)
        if prefix:
          tokens.append(prefix)
        index_str, remainder = suffix.split(']', 1)
        tokens.append(int(index_str))
        part = remainder[1:] if remainder.startswith('[') else remainder
      else:
        tokens.append(part)
        part = ''
  return tokens


def _extract(record: Any, expr: str) -> Any:
  current = record
  for token in _tokenize_field(expr):
    if current is None:
      return None
    if isinstance(token, int):
      if isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
        try:
          current = current[token]
        except IndexError:
          return None
      else:
        return None
    else:
      if isinstance(current, dict):
        current = current.get(token)
      else:
        return None
  return current


def _stringify(value: Any) -> Optional[str]:
  if value is None:
    return None
  if isinstance(value, (list, tuple)):
    if not value:
      return None
    candidate = value[0]
    if candidate is None:
      return None
    return str(candidate)
  return str(value)


def load_records(path: Path) -> Iterable[Record]:
  suffix = path.suffix.lower()
  if suffix in {'.jsonl', '.jsonl.gz'}:
    opener = path.open
    if suffix.endswith('.gz'):
      import gzip
      opener = lambda *args, **kwargs: gzip.open(path, 'rt', encoding='utf-8')
    with opener('r', encoding='utf-8') as handle:  # type: ignore[arg-type]
      for line in handle:
        if not line.strip():
          continue
        yield json.loads(line)
  elif suffix == '.json':
    data = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data, list):
      for item in data:
        yield item
    else:
      raise ValueError('JSON file must contain an array of objects')
  elif suffix == '.parquet':
    table = pq.read_table(path)
    for row in table.to_pylist():
      yield row
  elif suffix == '.arrow':
    with path.open('rb') as source:
      reader = pa_ipc.open_file(source)
      for batch in reader:
        for row in batch.to_pylist():
          yield row
  else:
    raise ValueError(f'Unsupported input format for {path}')


def build_prompt_response(record: Record, question_field: str, answer_field: str, context_field: Optional[str], prompt_template: str, response_template: str, skip_missing: bool) -> Optional[Dict[str, str]]:
  question = _stringify(_extract(record, question_field))
  answer = _stringify(_extract(record, answer_field))
  context = _stringify(_extract(record, context_field)) if context_field else None
  if skip_missing and (not question or not answer):
    return None
  data = {'question': question or '', 'answer': answer or ''}
  if context_field:
    data['context'] = context or ''
  prompt = prompt_template.format_map(data)
  response = response_template.format_map(data)
  return {'prompt': prompt, 'response': response}


def main() -> None:
  parser = argparse.ArgumentParser(description='Normalize QA dataset into prompt/response JSONL.')
  parser.add_argument('--input', required=True, type=Path, help='Input file (.json, .jsonl, .parquet, .arrow)')
  parser.add_argument('--output', required=True, type=Path, help='Destination JSONL file')
  parser.add_argument('--question-field', required=True, help='Field expression for question (supports dotted/indexed paths)')
  parser.add_argument('--answer-field', required=True, help='Field expression for answer (supports dotted/indexed paths)')
  parser.add_argument('--context-field', default=None, help='Optional field expression for additional context')
  parser.add_argument('--prompt-template', default='{question}', help='Template for prompt text')
  parser.add_argument('--response-template', default='{answer}', help='Template for response text')
  parser.add_argument('--num-threads', type=int, default=0, help='Number of threads to use (0 = library default)')
  parser.add_argument('--shuffle', action='store_true', help='Shuffle records before writing')
  parser.add_argument('--seed', type=int, default=1337, help='Random seed for shuffling')
  parser.add_argument('--max-records', type=int, default=None, help='Optional cap on number of records to emit')
  parser.add_argument('--skip-missing', action='store_true', help='Skip samples missing question/answer (default: False)')
  args = parser.parse_args()

  if args.num_threads > 0:
    os.environ['RAYON_NUM_THREADS'] = str(args.num_threads)
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

  records = list(load_records(args.input))
  if not records:
    raise ValueError('No records found in input dataset')
  if args.shuffle:
    random.Random(args.seed).shuffle(records)

  emitted = 0
  with args.output.open('w', encoding='utf-8') as handle:
    for record in records:
      item = build_prompt_response(record, args.question_field, args.answer_field, args.context_field, args.prompt_template, args.response_template, args.skip_missing)
      if item is None:
        continue
      handle.write(json.dumps(item, ensure_ascii=False) + '\n')
      emitted += 1
      if args.max_records is not None and emitted >= args.max_records:
        break
  print(f'Wrote {emitted} prompt/response pairs to {args.output}')


if __name__ == '__main__':
  main()
