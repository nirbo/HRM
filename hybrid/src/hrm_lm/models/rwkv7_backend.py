"""RWKV-7 encoder wrapper that plugs RWKV-PEFT modules into the HRM encoder stack."""  # Provide high-level module description

from __future__ import annotations  # Enable postponed evaluation of type hints for consistency

import os  # Manage environment variables required by RWKV kernels
import sys  # Allow dynamic path injection for local RWKV-PEFT checkout
from pathlib import Path  # Resolve filesystem paths relative to the repository
from typing import Optional, Tuple  # Provide type annotations for return values

import torch  # Core tensor library used by the encoder wrapper
import torch.nn as nn  # Neural network building blocks for the wrapper module

# Locate the vendored RWKV-PEFT repository that holds training-ready RWKV implementations
PEFT_ROOT = Path(__file__).resolve().parents[3] / "RWKV-PEFT"  # Derive absolute path to RWKV-PEFT clone at repo root
if PEFT_ROOT.exists():  # Confirm the repository clone is present before importing it
    if str(PEFT_ROOT) not in sys.path:  # Avoid duplicating entries when multiple imports occur
        sys.path.append(str(PEFT_ROOT))  # Expose RWKV-PEFT packages to Python import machinery
else:  # Handle missing clone explicitly to help users run setup script
    raise ImportError("RWKV-PEFT repository not found; run scripts/setup_rwkv_env.sh first")  # Provide actionable guidance when dependency is absent

from rwkvt.args_type import TrainingArgs  # Import training argument dataclass used to configure RWKV-7 modules
from rwkvt.rwkv7.model import RWKV7  # Import the actual RWKV-7 model definition from RWKV-PEFT


class RWKV7Encoder(nn.Module):  # Define encoder wrapper that exposes RWKV-7 through a Transformer-like interface
  def __init__(
    self,
    vocab_size: int,
    d_model: int,
    n_layers: int,
    max_seq_len: int,
    encoder_cfg: Optional[dict] = None,
  ) -> None:
    super().__init__()  # Initialize base nn.Module state
    cfg = encoder_cfg.copy() if encoder_cfg else {}  # Create a mutable configuration dictionary for local overrides
    if 'checkpoint_path' not in cfg or not cfg.get('checkpoint_path'):  # Detect absent checkpoint configuration
      default_ckpt = Path(__file__).resolve().parents[3] / 'models' / 'blinkdl-rwkv7-g1a-1.5b' / 'rwkv-final.pth'  # Resolve default RWKV-7 checkpoint location bundled by the user
      if default_ckpt.is_file():  # Only adopt default when the checkpoint actually exists locally
        cfg['checkpoint_path'] = str(default_ckpt)  # Inject default checkpoint path so integration works out of the box
    self._configure_environment(max_seq_len, cfg)  # Seed environment variables so RWKV kernels compile with correct parameters
    args = self._build_args(vocab_size, d_model, n_layers, max_seq_len, cfg)  # Generate TrainingArgs instance aligned with HRM configuration
    self.model = RWKV7(args)  # Instantiate RWKV-7 model using the assembled arguments
    checkpoint_path = Path(cfg.get('checkpoint_path', ''))  # Locate checkpoint path supplied by training config
    if not checkpoint_path.is_file():  # Validate that a checkpoint file is available before loading
      raise FileNotFoundError(f"RWKV-7 checkpoint missing at {checkpoint_path}")  # Provide informative error for missing weights
    state = torch.load(checkpoint_path, map_location='cpu')  # Load serialized parameter state onto CPU for deterministic initialization
    load_result = self.model.load_state_dict(state, strict=False)  # Populate RWKV-7 model weights while allowing head mismatches
    self._validate_load(load_result)  # Emit warnings when unexpected parameters are skipped or missing
    self.supports_cuda_graphs = False  # RWKV recurrence uses Triton/CUDA kernels that are not capture safe

  def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
  ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    token_embeddings = self.model.emb(input_ids)  # Embed token indices via RWKV-7 learned embedding table
    mask = self._prepare_mask(attention_mask, token_embeddings)  # Convert optional padding mask into broadcast-friendly tensor
    v_first = torch.zeros_like(token_embeddings)  # Initialize value cache placeholder consumed by the first block
    hidden = token_embeddings  # Seed hidden state with initial embeddings before block stack execution
    for block in self.model.blocks:  # Iterate through RWKV-7 blocks to accumulate sequential features
      hidden, v_first = block.forward_normal(hidden, v_first, attention_mask=mask)  # Execute attention + FFN block using provided mask
    hidden = self.model.ln_out(hidden)  # Apply final layer norm to produce stable hidden representations
    return hidden, None  # Expose hidden states and signal absence of auxiliary losses

  @staticmethod
  def _configure_environment(max_seq_len: int, cfg: dict) -> None:
    os.environ.setdefault('RWKV_MY_TESTING', str(cfg.get('model_variant', 'x070')))  # Select RWKV-7 kernel family by default
    os.environ['RWKV_CTXLEN'] = str(max_seq_len)  # Share context length so CUDA kernels compile with correct dimensions
    os.environ.setdefault('RWKV_HEAD_SIZE_A', str(cfg.get('head_size_a', 64)))  # Provide head size hint for kernel compilation
    os.environ.setdefault('RWKV_HEAD_SIZE_DIV', str(cfg.get('head_size_divisor', 8)))  # Mirror divisor used by RWKV-PEFT normalization
    os.environ.setdefault('RWKV_TRAIN_TYPE', str(cfg.get('train_type', 'none')))  # Default to standard training mode (no state tuning)
    os.environ.setdefault('WKV', str(cfg.get('wkv_backend', 'cuda')))  # Opt into CUDA custom kernels when available
    os.environ.setdefault('FUSED_KERNEL', '1' if cfg.get('fused_kernel', False) else '0')  # Enable fused kernels only when explicitly requested
    os.environ.setdefault('RWKV_FLOAT_MODE', str(cfg.get('float_mode', 'bf16')))  # Match precision hints expected by RWKV kernels

  @staticmethod
  def _build_args(
    vocab_size: int,
    d_model: int,
    n_layers: int,
    max_seq_len: int,
    cfg: dict,
  ) -> TrainingArgs:
    args = TrainingArgs()  # Start from RWKV-PEFT default argument template
    args.vocab_size = vocab_size  # Align vocabulary size with HRM configuration
    args.n_embd = d_model  # Match embedding dimension to HRM model width
    args.dim_att = cfg.get('dim_att', d_model)  # Configure attention dimension (defaults to embedding width)
    args.dim_ffn = cfg.get('dim_ffn', int((d_model * 3.5) // 32 * 32))  # Follow RWKV heuristic for FFN dimension alignment
    args.n_layer = n_layers  # Set number of RWKV blocks to requested depth
    args.ctx_len = max_seq_len  # Record maximum context length for kernel compilation
    args.head_size_a = cfg.get('head_size_a', 64)  # Configure per-head dimensionality for RWKV attention
    args.head_size_divisor = cfg.get('head_size_divisor', 8)  # Maintain divisor used by RWKV layer norm kernels
    args.my_testing = cfg.get('model_variant', 'x070')  # Tag variant to ensure correct kernel selection throughout the stack
    args.train_type = cfg.get('train_type', 'none')  # Respect optional training mode overrides (state / infctx)
    args.precision = cfg.get('precision', 'bf16')  # Store precision hint for downstream RWKV utilities
    return args  # Provide fully populated TrainingArgs structure to the caller

  @staticmethod
  def _validate_load(load_result: object) -> None:
    missing = getattr(load_result, 'missing_keys', [])  # Extract missing keys from load_state_dict result for diagnostics
    unexpected = getattr(load_result, 'unexpected_keys', [])  # Extract unexpected keys reported during state loading
    tolerated_missing = [key for key in missing if key == 'head.weight']  # Ignore missing output head because encoder does not use logits
    tolerated_unexpected = [key for key in unexpected if key == 'head.weight']  # Likewise ignore stray head weights in the checkpoint
    critical_missing = [key for key in missing if key not in tolerated_missing]  # Collect genuinely missing parameters
    critical_unexpected = [key for key in unexpected if key not in tolerated_unexpected]  # Collect genuinely unexpected parameters
    if critical_missing or critical_unexpected:  # Raise when critical discrepancies remain after filtering tolerated keys
      raise RuntimeError(f'RWKV-7 checkpoint mismatch: missing {critical_missing}, unexpected {critical_unexpected}')  # Abort initialization with descriptive error

  @staticmethod
  def _prepare_mask(
    attention_mask: Optional[torch.Tensor],
    token_embeddings: torch.Tensor,
  ) -> Optional[torch.Tensor]:
    if attention_mask is None:  # Skip processing when no mask is supplied
      return None  # Propagate absence upstream to signal lack of padding information
    if attention_mask.dtype == torch.bool:  # Detect boolean padding masks generated by HRM trainer
      mask = (~attention_mask).to(dtype=token_embeddings.dtype, device=token_embeddings.device)  # Invert pad mask so valid tokens become one
    else:  # Handle numeric masks that may already encode validity scores
      mask = attention_mask.to(dtype=token_embeddings.dtype, device=token_embeddings.device)  # Cast numeric masks to embedding dtype/device without inversion
    return mask  # Provide broadcast-ready mask with ones at valid token positions
