"""Tokenization and synthetic data helpers for HRM-LM integration."""  # module docstring

from .simple_tokenizer import PAD_ID, BOS_ID, EOS_ID, SimpleTokenizer  # re-export tokenizer
from .synthetic import build_synthetic_dataset, pad_batch  # expose dataset utilities

__all__ = ['PAD_ID', 'BOS_ID', 'EOS_ID', 'SimpleTokenizer', 'build_synthetic_dataset', 'pad_batch']  # controlled exports
