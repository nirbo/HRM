from __future__ import annotations

"""Build a byte-level text dataset for HRM from JSONL files.

Input format (per line JSON): {"prompt": "...", "target": "..."}
Output format matches other datasets in this repo.
"""

import json
import os
from typing import Optional

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

from dataset.common import PuzzleDatasetMetadata


cli = ArgParser()


class TextDataConfig(BaseModel):
    input_dir: str = "data/raw/text_tasks"
    output_dir: str = "data/text-tasks"
    seq_len: int = 512

    # optional subsample for quick tests
    subsample_size: Optional[int] = None


def _encode_bytes(s: str) -> np.ndarray:
    b = s.encode("utf-8", errors="replace")
    arr = np.frombuffer(b, dtype=np.uint8).astype(np.int32) + 1  # 1..256
    return arr


def _load_split(path: str, seq_len: int, subsample: Optional[int]):
    inputs, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"reading {os.path.basename(path)}"):
            obj = json.loads(line)
            prompt = obj.get("prompt", "")
            target = obj.get("target", "")
            x = _encode_bytes(prompt)
            y = _encode_bytes(target)
            # fit to seq_len
            x = x[:seq_len]
            y = y[:seq_len]
            if x.size < seq_len:
                x = np.pad(x, (0, seq_len - x.size))
            if y.size < seq_len:
                y = np.pad(y, (0, seq_len - y.size))
            inputs.append(x)
            labels.append(y)

    if subsample is not None and len(inputs) > subsample:
        idx = np.random.permutation(len(inputs))[:subsample]
        inputs = [inputs[i] for i in idx]
        labels = [labels[i] for i in idx]

    # pack single-example puzzles, one per group
    n = len(inputs)
    puzzle_identifiers = np.zeros((n,), dtype=np.int32)
    puzzle_indices = np.arange(n + 1, dtype=np.int32)
    group_indices = np.arange(n + 1, dtype=np.int32)

    results = {
        "inputs": np.stack(inputs).astype(np.int32),
        "labels": np.stack(labels).astype(np.int32),
        "puzzle_identifiers": puzzle_identifiers,
        "puzzle_indices": puzzle_indices,
        "group_indices": group_indices,
    }
    return results


def _save_split(out_dir: str, set_name: str, results):
    os.makedirs(os.path.join(out_dir, set_name), exist_ok=True)
    # Metadata
    meta = PuzzleDatasetMetadata(
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        vocab_size=256 + 1,
        seq_len=results["inputs"].shape[1],
        num_puzzle_identifiers=1,
        total_groups=results["group_indices"].size - 1,
        mean_puzzle_examples=1.0,
        sets=["all"],
    )
    with open(os.path.join(out_dir, set_name, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(meta.model_dump(), f)
    for k, v in results.items():
        np.save(os.path.join(out_dir, set_name, f"all__{k}.npy"), v)


@cli.command(singleton=True)
def build(config: TextDataConfig):
    train_path = os.path.join(config.input_dir, "train.jsonl")
    test_path = os.path.join(config.input_dir, "test.jsonl")

    train_res = _load_split(train_path, config.seq_len, config.subsample_size)
    test_res = _load_split(test_path, config.seq_len, None)

    _save_split(config.output_dir, "train", train_res)
    _save_split(config.output_dir, "test", test_res)


if __name__ == "__main__":
    cli()

