#!/usr/bin/env python
"""Merge chunked HRM-LM dataset batches into a single train/val pair.

This module exposes a CLI that consumes the intermediate directories produced by
chunked invocations of ``prepare_language_dataset.py``.  Each chunk directory is
expected to contain ``train.jsonl``, ``val.jsonl``, and ``meta.json`` files.  The
script streams those chunk files into consolidated outputs without ever holding
an entire shard in memory, ensuring disk usage stays roughly constant during the
merge.  After successful consolidation the source chunk files (and directories)
are removed unless the ``--keep-chunks`` flag is supplied.  The script persists a
fresh ``meta.json`` alongside the consolidated outputs that aggregates sample
counts and carries forward tokenizer metadata.

Functions:
  - ``append_stream``: copy a source file into an already-open destination
    handle using a bounded buffer to limit memory usage.
  - ``load_meta``: read and validate a chunk ``meta.json`` payload.
  - ``merge_batches``: orchestrate the end-to-end merge workflow.

Dependencies:
  - Standard library only (``argparse``, ``json``, ``os``, ``pathlib``, ``shutil``).
  - No external modules are required beyond the Python runtime.

Integration:
  - Invoked after chunk preprocessing completes to collapse the chunk output
    tree into a single dataset that training jobs can consume directly.
  - Designed to be called from shell wrappers or orchestration scripts once all
    parallel chunk conversions finish.
"""

import argparse  # Parse command-line flags for batch/root/output configuration.
import json  # Load and emit metadata payloads for each dataset chunk.
import os  # Provide low-level OS helpers used during atomic file replacement.
from pathlib import Path  # Handle filesystem paths with convenient methods.
from typing import Dict, Iterable, Optional, Tuple  # Annotate helper function signatures.

import shutil  # Stream chunk files into consolidated outputs with copyfileobj.

BUFFER_SIZE = 4 * 1024 * 1024  # Use a 4 MiB buffer to balance throughput and memory.


def append_stream(source: Path, destination) -> int:
  """Stream ``source`` into ``destination`` and return the number of bytes copied."""
  total_bytes = 0  # Track the cumulative bytes transferred for diagnostics.
  with source.open('rb') as src_file:  # Open the source chunk for binary reading.
    while True:  # Iterate until the entire file has been streamed.
      chunk = src_file.read(BUFFER_SIZE)  # Read up to BUFFER_SIZE bytes from the source.
      if not chunk:  # Exit once the read returns no additional data.
        break  # Terminate the streaming loop when the source is exhausted.
      destination.write(chunk)  # Append the binary block to the destination handle.
      total_bytes += len(chunk)  # Update the byte counter with the chunk size.
  return total_bytes  # Report the transfer size to the caller.


def load_meta(meta_path: Path) -> Dict[str, int]:
  """Load a chunk ``meta.json`` file and return its dictionary payload."""
  with meta_path.open('r', encoding='utf-8') as handle:  # Open metadata as UTF-8 text.
    return json.load(handle)  # Decode JSON content into a dictionary structure.


def merge_batches(batch_root: Path, output_dir: Path, keep_chunks: bool) -> None:
  """Merge every chunk directory inside ``batch_root`` into ``output_dir``."""
  if not batch_root.exists():  # Ensure the batch root exists before proceeding.
    raise FileNotFoundError(f'Batch directory {batch_root} does not exist')  # Fail fast on missing input.
  chunk_dirs = sorted([path for path in batch_root.iterdir() if path.is_dir()])  # Collect chunk directories in lexical order.
  if not chunk_dirs:  # Guard against empty batch roots.
    raise ValueError(f'No chunk directories found in {batch_root}')  # Signal invalid invocation when nothing to merge.

  output_dir.mkdir(parents=True, exist_ok=True)  # Create the destination directory tree if needed.
  train_tmp = output_dir / 'train.jsonl.tmp'  # Define a temporary path for the consolidated train file.
  val_tmp = output_dir / 'val.jsonl.tmp'  # Define a temporary path for the consolidated validation file.
  train_out = output_dir / 'train.jsonl'  # Final consolidated train dataset path.
  val_out = output_dir / 'val.jsonl'  # Final consolidated validation dataset path.
  meta_out = output_dir / 'meta.json'  # Final consolidated metadata path.

  if train_out.exists() or val_out.exists():  # Prevent accidental overwrite of existing aggregated outputs.
    raise FileExistsError('Consolidated dataset already exists; remove it or pick another output directory')  # Warn user to avoid data loss.

  total_train = 0  # Initialize aggregate train sample counter.
  total_val = 0  # Initialize aggregate validation sample counter.
  max_seq_len: Optional[int] = None  # Track sequence length metadata from chunks.
  vocab_size: Optional[int] = None  # Track tokenizer vocabulary size metadata from chunks.
  tokenizer_file: Optional[str] = None  # Track tokenizer path metadata from chunks.

  if train_tmp.exists():  # Remove stale temporary artifacts from previous runs if present.
    train_tmp.unlink()  # Delete existing temporary train file to start cleanly.
  if val_tmp.exists():  # Check for leftover validation temporary data.
    val_tmp.unlink()  # Remove stale temporary validation file contents.

  with train_tmp.open('ab') as train_handle, val_tmp.open('ab') as val_handle:  # Open temporary outputs for appending.
    for chunk_dir in chunk_dirs:  # Process each chunk directory sequentially to control disk usage.
      chunk_train = chunk_dir / 'train.jsonl'  # Path to the chunk train file.
      chunk_val = chunk_dir / 'val.jsonl'  # Path to the chunk validation file.
      chunk_meta = chunk_dir / 'meta.json'  # Path to the chunk metadata file.
      if not chunk_train.exists() or not chunk_val.exists() or not chunk_meta.exists():  # Verify chunk completeness.
        raise FileNotFoundError(f'Chunk {chunk_dir} is missing expected files')  # Halt if a required file is absent.

      meta = load_meta(chunk_meta)  # Parse chunk metadata for aggregation stats.
      total_train += int(meta.get('train_samples', 0))  # Accumulate train sample count from the current chunk.
      total_val += int(meta.get('val_samples', 0))  # Accumulate validation sample count from the current chunk.

      if max_seq_len is None and 'max_seq_len' in meta:  # Capture the maximum sequence length from the first chunk that specifies it.
        max_seq_len = int(meta['max_seq_len'])  # Store sequence length metadata for final meta output.
      if vocab_size is None and 'vocab_size' in meta:  # Capture vocabulary size metadata if provided.
        vocab_size = int(meta['vocab_size'])  # Store vocabulary size for future reference.
      if tokenizer_file is None and 'tokenizer_file' in meta:  # Capture tokenizer file path metadata if provided.
        tokenizer_file = str(meta['tokenizer_file'])  # Store tokenizer path to include in final metadata.

      append_stream(chunk_train, train_handle)  # Append the chunk train data to the consolidated train stream.
      append_stream(chunk_val, val_handle)  # Append the chunk validation data to the consolidated validation stream.

      if not keep_chunks:  # Optionally clean up chunk artifacts to reclaim disk space.
        chunk_train.unlink()  # Remove the chunk train file.
        chunk_val.unlink()  # Remove the chunk validation file.
        chunk_meta.unlink()  # Remove the chunk metadata file.
        try:  # Attempt to remove the chunk directory now that its files are gone.
          chunk_dir.rmdir()  # Remove the now-empty chunk directory.
        except OSError:  # Ignore removal failures when residual files still exist.
          # Leave the directory in place if concurrent processes temporarily hold files.
          pass  # Skip removal so that future cleanup can retry once the directory is free.

  os.replace(train_tmp, train_out)  # Atomically promote the train temp file to its final location.
  os.replace(val_tmp, val_out)  # Atomically promote the validation temp file to its final location.

  consolidated_meta = {  # Build the aggregate metadata dictionary for the consolidated dataset.
    'train_samples': total_train,  # Total number of train samples across all chunks.
    'val_samples': total_val,  # Total number of validation samples across all chunks.
    'max_seq_len': max_seq_len if max_seq_len is not None else 0,  # Propagate sequence length metadata or default to 0.
    'vocab_size': vocab_size if vocab_size is not None else 0,  # Propagate vocabulary size metadata or default to 0.
    'tokenizer_file': tokenizer_file if tokenizer_file is not None else '',  # Propagate tokenizer file path metadata if available.
  }

  with meta_out.open('w', encoding='utf-8') as meta_handle:  # Open the consolidated meta output for writing.
    json.dump(consolidated_meta, meta_handle, indent=2)  # Persist the aggregated metadata with indentation for readability.
    meta_handle.write('\n')  # Ensure the metadata file ends with a newline character.

  if not keep_chunks and not any(batch_root.iterdir()):  # Optionally tidy up the batch root when it becomes empty.
    batch_root.rmdir()  # Remove the now-empty batch root directory.


if __name__ == '__main__':  # Execute CLI logic when the module is invoked directly.
  parser = argparse.ArgumentParser(description='Merge chunked HRM-LM dataset outputs into a single dataset.')  # Set up CLI parser with descriptive help text.
  parser.add_argument('--batches', type=Path, required=True, help='Directory containing chunk subdirectories to merge.')  # Define required batch root argument.
  parser.add_argument('--output-dir', type=Path, required=True, help='Destination directory for the consolidated dataset.')  # Define required output directory argument.
  parser.add_argument('--keep-chunks', action='store_true', help='Preserve chunk directories instead of deleting them after merge.')  # Optional flag to retain chunk artifacts.
  args = parser.parse_args()  # Parse command-line arguments into a namespace.

  merge_batches(batch_root=args.batches, output_dir=args.output_dir, keep_chunks=args.keep_chunks)  # Invoke the merge workflow with parsed arguments.
