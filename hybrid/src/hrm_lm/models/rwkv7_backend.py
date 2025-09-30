"""RWKV-7 encoder wrapper that plugs RWKV-PEFT modules into the HRM encoder stack."""  # Provide high-level module description

from __future__ import annotations  # Enable postponed evaluation of type hints for consistency

import logging  # Provide module-level logger for backend selection visibility
import os  # Manage environment variables required by RWKV kernels
import subprocess  # Invoke external build tools required by RWKV kernels
import sys  # Allow dynamic path injection for local repository clones
import warnings  # Emit runtime warnings when kernel fallbacks are triggered
from pathlib import Path  # Resolve filesystem paths relative to the repository
from typing import Callable, Dict, List, Optional, Tuple  # Provide type annotations for return values

import torch  # Core tensor library used by the encoder wrapper
import torch.nn as nn  # Neural network building blocks for the wrapper module
import torch.utils.cpp_extension as torch_cpp_extension  # Access cpp extension helpers for monkey patches

logger = logging.getLogger(__name__)  # Provide module-level logger for backend selection visibility


# Default RWKV env vars must be present before importing RWKV-PEFT modules.
os.environ.setdefault('FUSED_KERNEL', '0')  # Disable fused kernels unless explicitly enabled
os.environ.setdefault('RWKV_HEAD_SIZE_A', '64')  # Provide default head size for kernel compilation
os.environ.setdefault('RWKV_HEAD_SIZE_DIV', '8')  # Provide default divisor for group norm kernels
os.environ.setdefault('RWKV_TRAIN_TYPE', 'none')  # Default to standard training mode
os.environ.setdefault('RWKV_MY_TESTING', 'x070')  # Default variant tag for RWKV-7 kernels
os.environ.setdefault('WKV', 'cuda')  # Use CUDA kernels by default
torch_cpp_extension.verify_ninja_availability = lambda: None  # Bypass strict ninja presence check; we provide path below

# Ensure ninja binary from the active virtualenv is discoverable by torch extensions.
venv_bin = Path(sys.executable).resolve().parent
os.environ['PATH'] = f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"
os.putenv('PATH', os.environ['PATH'])
ninja_exe = venv_bin / 'ninja'
if ninja_exe.exists():
  os.environ['NINJA_EXECUTABLE'] = str(ninja_exe)
  os.environ['NINJA'] = str(ninja_exe)
  assert os.environ['NINJA_EXECUTABLE']

_orig_run_ninja = torch_cpp_extension._run_ninja_build

def _patched_run_ninja_build(build_directory: str, verbose: bool, error_prefix: str) -> None:  # pragma: no cover - mirrors upstream build behaviour
  ninja_path = Path(sys.executable).parent / 'ninja'
  command = [str(ninja_path) if ninja_path.exists() else 'ninja', '-v']
  num_workers = torch_cpp_extension._get_num_workers(verbose)
  if num_workers is not None:
    command.extend(['-j', str(num_workers)])
  env = os.environ.copy()
  if torch_cpp_extension.IS_WINDOWS and 'VSCMD_ARG_TGT_ARCH' not in env:
    from setuptools import distutils  # type: ignore[attr-defined]

    plat_name = distutils.util.get_platform()
    plat_spec = torch_cpp_extension.PLAT_TO_VCVARS[plat_name]
    vc_env = {k.upper(): v for k, v in torch_cpp_extension._get_vc_env(plat_spec).items()}
    for k, v in env.items():
      uk = k.upper()
      if uk not in vc_env:
        vc_env[uk] = v
    env = vc_env
  try:
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fileno = 1
    subprocess.run(
      command,
      shell=torch_cpp_extension.IS_WINDOWS and torch_cpp_extension.IS_HIP_EXTENSION,
      stdout=stdout_fileno if verbose else subprocess.PIPE,
      stderr=subprocess.STDOUT,
      cwd=build_directory,
      check=True,
      env=env,
    )
  except subprocess.CalledProcessError as e:
    _, error, _ = sys.exc_info()
    message = error_prefix
    if hasattr(error, 'output') and error.output:  # type: ignore[union-attr]
      message += f": {error.output.decode(*torch_cpp_extension.SUBPROCESS_DECODE_ARGS)}"  # type: ignore[union-attr]
    raise RuntimeError(message) from e

torch_cpp_extension._run_ninja_build = _patched_run_ninja_build

# Locate the vendored RWKV-PEFT repository that holds training-ready RWKV implementations
PEFT_ROOT = Path(__file__).resolve().parents[3] / "RWKV-PEFT"  # Derive absolute path to RWKV-PEFT clone at repo root
if PEFT_ROOT.exists():  # Confirm the repository clone is present before importing it
    if str(PEFT_ROOT) not in sys.path:  # Avoid duplicating entries when multiple imports occur
        sys.path.append(str(PEFT_ROOT))  # Expose RWKV-PEFT packages to Python import machinery
else:  # Handle missing clone explicitly to help users run setup script
    raise ImportError("RWKV-PEFT repository not found; run scripts/setup_rwkv_env.sh first")  # Provide actionable guidance when dependency is absent

WIND_ROOT = Path(__file__).resolve().parents[3] / 'wind_rwkv'  # Locate optional wind_rwkv clone providing tuned kernels
if WIND_ROOT.exists():  # Only modify sys.path when repository is present
    if str(WIND_ROOT) not in sys.path:  # Prevent duplicate sys.path entries on repeated imports
        sys.path.append(str(WIND_ROOT))  # Allow importing wind_rwkv without pip installing
FLA_ROOT = Path(__file__).resolve().parents[3] / 'flash-linear-attention'  # Locate flash-linear-attention clone for FLA kernels
if FLA_ROOT.exists():  # Guard to avoid path pollution if clone is missing
    if str(FLA_ROOT) not in sys.path:  # Avoid injecting duplicate entries
        sys.path.append(str(FLA_ROOT))  # Expose fla Python package to import hooks

_cwd = os.getcwd()
os.chdir(str(PEFT_ROOT))
try:
  from rwkvt.args_type import TrainingArgs  # Import training argument dataclass used to configure RWKV-7 modules
  from rwkvt.rwkv7.model import RWKV7  # Import the actual RWKV-7 model definition from RWKV-PEFT
finally:
  os.chdir(_cwd)


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
    self.kernel_backend = self._select_backend(args, cfg)  # Determine and activate the most efficient kernel backend before loading checkpoints
    checkpoint_path = Path(cfg.get('checkpoint_path', ''))  # Locate checkpoint path supplied by training config
    if not checkpoint_path.is_file():  # Validate that a checkpoint file is available before loading
      raise FileNotFoundError(f"RWKV-7 checkpoint missing at {checkpoint_path}")  # Provide informative error for missing weights
    state = torch.load(checkpoint_path, map_location='cpu')  # Load serialized parameter state onto CPU for deterministic initialization
    load_result = self.model.load_state_dict(state, strict=False)  # Populate RWKV-7 model weights while allowing head mismatches
    self._validate_load(load_result)  # Emit warnings when unexpected parameters are skipped or missing
    self.supports_cuda_graphs = False  # RWKV recurrence uses Triton/CUDA kernels that are not capture safe

  def _select_backend(self, args: TrainingArgs, cfg: dict) -> str:  # Decide which RWKV kernel implementation to activate
    preference = str(cfg.get('kernel_preference', 'auto')).lower()  # Normalize kernel preference override from configuration
    device_info = self._describe_device()  # Capture accelerator characteristics used for capability checks
    candidate_map: Dict[str, Callable[[TrainingArgs, dict, Dict[str, object]], None]] = {  # Map backend keys to activation routines
      'wind_chunked': self._activate_wind_chunked,  # Fast NVIDIA kernel path
      'wind_longhead': self._activate_wind_longhead,  # General-purpose CUDA/HIP kernel
      'fla_chunk': self._activate_fla_chunk,  # Flash Linear Attention fallback
    }
    if preference == 'default':  # Honor explicit request to keep stock kernels
      logger.info('Using default RWKV-PEFT kernels per configuration override')  # Emit informational log for transparency
      return 'default'  # Signal that no patching occurred
    if preference != 'auto':  # Handle explicit kernel selection
      if preference not in candidate_map:  # Validate that the requested key exists
        raise ValueError(f"Unknown RWKV kernel preference '{preference}'")  # Surface configuration mistakes early
      order: List[str] = [preference]  # Restrict candidate order to the requested backend
    else:  # Automatic selection branch
      order = self._auto_candidate_order(args, cfg, device_info)  # Build best-effort priority list based on hardware
    errors: List[str] = []  # Collect failure reasons to report if all candidates fail
    for candidate in order:  # Iterate through backend options
      if candidate == 'default':  # Default entry acts as a sentinel
        logger.info('Falling back to default RWKV-PEFT kernels')  # Record fallback decision
        return 'default'  # Expose fallback outcome
      try:
        candidate_map[candidate](args, cfg, device_info)  # Attempt to activate the selected backend
        logger.info('Activated RWKV backend %s', candidate)  # Record the successful backend
        return candidate  # Expose the backend identifier to callers
      except Exception as exc:  # Catch activation failures
        errors.append(f"{candidate}: {exc}")  # Accumulate concise error for later warning
        logger.debug('Kernel backend %s failed during activation', candidate, exc_info=exc)  # Provide stack trace for debugging when verbose logging is enabled
    if errors:  # Only warn when at least one backend failed
      warnings.warn(  # Emit aggregated warning so users can diagnose why auto selection degraded
        'All preferred RWKV kernel backends failed; using default kernels. Reasons: ' + '; '.join(errors),
        RuntimeWarning,
        stacklevel=2,
      )
    logger.info('Defaulting to RWKV-PEFT kernels after backend activation attempts')  # Confirm final fallback
    return 'default'  # Return fallback indicator

  def _auto_candidate_order(self, args: TrainingArgs, cfg: dict, device_info: Dict[str, object]) -> List[str]:  # Build backend priority order for auto mode
    if not device_info.get('is_cuda', False):  # Abort early when no GPU accelerator is available
      return ['default']  # CPU-only execution relies on stock kernels
    order: List[str] = []  # Initialize candidate list
    float_mode = str(cfg.get('float_mode', 'bf16')).lower()  # Read configured precision to ensure kernel compatibility
    ctx_multiple_ok = args.ctx_len % 16 == 0  # Chunked kernels require sequence lengths divisible by 16
    if (
      device_info.get('is_nvidia', False)  # NVIDIA GPU detected
      and device_info.get('sm', 0) >= 80  # Compute capability SM80 or newer
      and float_mode in ('bf16', 'bfloat16')  # Chunked kernel expects bfloat16 math
      and ctx_multiple_ok  # Ensure context length satisfies chunk size constraint
    ):
      order.append('wind_chunked')  # Prefer chunked CUDA kernel when all constraints hold
    order.append('wind_longhead')  # Next best option handles larger heads and cross-vendor GPUs
    order.append('fla_chunk')  # Flash Linear Attention fallback covers broad hardware
    order.append('default')  # Always terminate with default sentinel
    return order  # Provide ordered backend preference list

  @staticmethod
  def _describe_device() -> Dict[str, object]:  # Gather accelerator metadata used for backend gating
    info: Dict[str, object] = {'is_cuda': torch.cuda.is_available()}  # Record CUDA availability upfront
    if not info['is_cuda']:  # Exit early when no GPU exists
      return info  # Minimal information suffices for fallback decisions
    index = torch.cuda.current_device()  # Capture active device index
    name = torch.cuda.get_device_name(index)  # Query device marketing name for vendor detection
    capability = torch.cuda.get_device_capability(index)  # Retrieve compute capability tuple (major, minor)
    sm = capability[0] * 10 + capability[1]  # Convert capability tuple into SM-style integer for comparisons
    info.update(  # Populate additional metadata fields
      {
        'index': index,  # Persist device index for troubleshooting
        'name': name,  # Persist human-readable GPU name
        'capability': capability,  # Expose raw capability tuple as reference
        'sm': sm,  # Store flattened SM score for threshold comparisons
        'is_nvidia': 'NVIDIA' in name.upper(),  # Flag NVIDIA hardware based on name pattern
        'is_amd': 'AMD' in name.upper() or torch.version.hip is not None,  # Flag AMD/ROCm hardware heuristically
      }
    )
    return info  # Provide complete device description

  def _activate_wind_chunked(self, args: TrainingArgs, cfg: dict, device_info: Dict[str, object]) -> None:  # Enable wind chunked CUDA kernel
    if not device_info.get('is_nvidia', False):  # Guard against unsupported vendors
      raise RuntimeError('wind chunked kernel requires an NVIDIA GPU')  # Communicate incompatibility clearly
    if device_info.get('sm', 0) < 80:  # Enforce SM80+ requirement for chunked kernel instructions
      raise RuntimeError('wind chunked kernel requires SM80 or newer hardware')  # Explain why activation failed
    if args.ctx_len % 16 != 0:  # Validate sequence divisibility requirement
      raise RuntimeError('context length must be a multiple of 16 for wind chunked kernel')  # Provide actionable guidance
    float_mode = str(cfg.get('float_mode', 'bf16')).lower()  # Inspect configured precision
    if float_mode not in ('bf16', 'bfloat16'):  # wind chunked asserts on dtype mismatch
      raise RuntimeError('wind chunked kernel expects bfloat16 precision')  # Clarify failure when precision mismatches
    try:
      from wind_rwkv.rwkv7.chunked_cuda.chunked_cuda import attn_chunked_cuda, load_chunked_cuda  # Import kernel loader lazily
    except ImportError as exc:  # Surface missing dependency
      raise RuntimeError('wind_rwkv chunked kernels are unavailable; run setup script to clone/install wind_rwkv') from exc  # Suggest remediation
    load_chunked_cuda(args.head_size_a)  # Compile/load kernel for the configured head size
    from rwkvt.operator import rwkvop  # Import operator module after kernel load so we can patch function pointer
    head_size = args.head_size_a  # Cache head size locally for closure capture

    def run_chunked(r: torch.Tensor, w: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a: torch.Tensor, b: torch.Tensor, HEAD_SIZE: int = head_size) -> torch.Tensor:  # Define runtime bridge into wind kernel
      B, T, HC = w.shape  # Unpack tensor dimensions
      if HC % HEAD_SIZE != 0:  # Validate divisibility between hidden channels and head size
        raise RuntimeError('hidden size is not divisible by RWKV head size for chunked kernel')  # Abort with descriptive error
      H = HC // HEAD_SIZE  # Compute number of heads dynamically
      shaped = [tensor.contiguous().view(B, T, H, HEAD_SIZE) for tensor in (r, w, k, v, a, b)]  # Reshape inputs to kernel layout
      outputs, _ = attn_chunked_cuda(*shaped)  # Execute forward pass and discard state output
      return outputs.view(B, T, HC)  # Restore original flattened layout

    rwkvop.RUN_CUDA_RWKV7g = run_chunked  # Patch RWKV operator to use wind chunked kernel

  def _activate_wind_longhead(self, args: TrainingArgs, cfg: dict, device_info: Dict[str, object]) -> None:  # Enable wind longhead kernel for broad hardware
    if not device_info.get('is_cuda', False):  # Ensure a CUDA/HIP device is present
      raise RuntimeError('wind longhead kernel requires CUDA or HIP runtime')  # Communicate missing accelerator
    float_mode = str(cfg.get('float_mode', 'bf16')).lower()  # Read precision configuration
    if float_mode not in ('bf16', 'bfloat16'):  # Kernel asserts on bf16 inputs
      raise RuntimeError('wind longhead kernel expects bfloat16 precision')  # Provide explicit failure reason
    try:
      from wind_rwkv.rwkv7.backstepping_longhead.backstepping_longhead import attn_backstepping_longhead, load_backstepping_longhead  # Import longhead kernel utilities
    except ImportError as exc:  # Missing dependency path
      raise RuntimeError('wind_rwkv longhead kernels are unavailable; ensure wind_rwkv is cloned') from exc  # Encourage environment setup
    load_backstepping_longhead(args.head_size_a)  # Compile/load kernel for current head size
    from rwkvt.operator import rwkvop  # Access operator patch point
    head_size = args.head_size_a  # Capture head size for closure

    def run_longhead(r: torch.Tensor, w: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a: torch.Tensor, b: torch.Tensor, HEAD_SIZE: int = head_size) -> torch.Tensor:  # Define bridge for longhead kernel
      B, T, HC = w.shape  # Read tensor dimensions
      if HC % HEAD_SIZE != 0:  # Confirm divisibility
        raise RuntimeError('hidden size is not divisible by RWKV head size for longhead kernel')  # Provide error context
      H = HC // HEAD_SIZE  # Compute number of heads
      shaped = [tensor.contiguous().view(B, T, H, HEAD_SIZE) for tensor in (r, w, k, v, a, b)]  # Reshape into kernel expected format
      outputs, _ = attn_backstepping_longhead(*shaped)  # Execute kernel and ignore final state output
      return outputs.view(B, T, HC)  # Flatten back to original layout

    rwkvop.RUN_CUDA_RWKV7g = run_longhead  # Switch RWKV operator to wind longhead implementation

  def _activate_fla_chunk(self, args: TrainingArgs, cfg: dict, device_info: Dict[str, object]) -> None:  # Enable Flash Linear Attention fallback kernel
    if not device_info.get('is_cuda', False):  # FLA kernels rely on GPU execution
      raise RuntimeError('flash-linear-attention fallback requires a CUDA/HIP device')  # Clarify requirement
    try:
      from fla.ops.rwkv7.chunk import chunk_rwkv7  # Import FLA chunk kernel lazily
    except ImportError as exc:  # Missing dependency path
      raise RuntimeError('flash-linear-attention repository not available; ensure it is cloned/installed') from exc  # Point to setup fix
    from rwkvt.operator import rwkvop  # Import operator patch point
    head_size = args.head_size_a  # Capture head size for view reshaping

    def run_fla(r: torch.Tensor, w: torch.Tensor, k: torch.Tensor, v: torch.Tensor, a: torch.Tensor, b: torch.Tensor, HEAD_SIZE: int = head_size) -> torch.Tensor:  # Define bridge to FLA kernel
      B, T, HC = w.shape  # Capture dimensions
      if HC % HEAD_SIZE != 0:  # Validate divisibility by head size
        raise RuntimeError('hidden size is not divisible by RWKV head size for FLA kernel')  # Provide context on mismatch
      H = HC // HEAD_SIZE  # Compute number of attention heads
      shaped = [tensor.contiguous().view(B, T, H, HEAD_SIZE) for tensor in (r, w, k, v, a, b)]  # Reshape to [B, T, H, head]
      outputs, _ = chunk_rwkv7(  # Execute FLA chunk kernel ignoring final state output
        r=shaped[0],
        w=shaped[1],
        k=shaped[2],
        v=shaped[3],
        a=shaped[4],
        b=shaped[5],
        scale=1.0,
        initial_state=None,
        output_final_state=False,
        head_first=False,
      )
      return outputs.view(B, T, HC)  # Restore flattened tensor layout for downstream modules

    rwkvop.RUN_CUDA_RWKV7g = run_fla  # Patch RWKV operator to leverage FLA kernel

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
