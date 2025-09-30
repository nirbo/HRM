"""RWKV-7 encoder wrapper that plugs RWKV-PEFT modules into the HRM encoder stack."""  # Provide high-level module description

from __future__ import annotations  # Enable postponed evaluation of type hints for consistency

import logging  # Provide module-level logger for backend selection visibility
import os  # Manage environment variables required by RWKV kernels
import subprocess  # Invoke external build tools required by RWKV kernels
import sys  # Allow dynamic path injection for local repository clones
import warnings  # Emit runtime warnings when kernel fallbacks are triggered
from copy import deepcopy  # Clone global PEFT templates without mutating upstream defaults
from pathlib import Path  # Resolve filesystem paths relative to the repository
from typing import Any, Callable, Dict, List, Optional, Tuple  # Provide type annotations for return values

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

os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')  # Disable torch.compile ahead of RWKV module imports to avoid Dynamo assertions
os.environ.setdefault('DISABLE_TORCH_COMPILE', '1')  # Secondary flag recognised by some runtimes

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
  from rwkvt.peft.rwkvLinear import DiSHA_CONFIG, LORA_CONFIG  # Import global PEFT configuration structures used to enable adapters
finally:
  os.chdir(_cwd)


class RWKV7Encoder(nn.Module):  # Define encoder wrapper that exposes RWKV-7 through a Transformer-like interface
  @staticmethod
  def _patch_att_kernel(func: Callable[..., torch.Tensor]) -> None:
    try:
      import importlib
      att_mod = importlib.import_module('rwkvt.rwkv7.att')
    except Exception:
      return
    setattr(att_mod, 'RUN_CUDA_RWKV7g', func)

  @staticmethod
  def _to_plain_dict(value: Optional[Any]) -> Dict[str, Any]:  # Normalise nested config mappings without requiring OmegaConf globally
    if value is None:
      return {}
    if isinstance(value, dict):
      return dict(value)
    try:  # Avoid hard OmegaConf dependency when not installed
      from omegaconf import DictConfig, OmegaConf  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
      DictConfig = None  # type: ignore
      OmegaConf = None  # type: ignore
    if 'DictConfig' in locals() and DictConfig is not None and isinstance(value, DictConfig):  # type: ignore
      assert OmegaConf is not None  # narrow type for mypy
      return dict(OmegaConf.to_container(value, resolve=True))
    if hasattr(value, 'items'):
      return dict(value)  # type: ignore[arg-type]
    return {}

  @classmethod
  def _normalise_peft_config(cls, raw_cfg: Optional[Any]) -> Dict[str, Any]:  # Convert arbitrary user config into canonical PEFT settings
    base_lora = {
      'r': 0,
      'alpha': 0.0,
      'dropout': 0.0,
      'target_parts': ['att', 'ffn'],
      'load_path': '',
    }
    base_pissa = {
      'r': 0,
      'svd_niter': 4,
      'target_parts': ['att', 'ffn'],
      'load_path': '',
      'init_path': '',
    }
    base_disha = {
      'mode': 'bone',
      'r': 0,
      'target_parts': ['att', 'ffn'],
      'load_path': '',
    }
    cfg: Dict[str, Any] = {
      'type': 'none',
      'freeze_non_peft': True,
      'train_embeddings': True,
      'train_head': True,
      'train_layer_norms': True,
      'train_parts': ['time', 'ln'],
      'quantization': 'none',
      'lora': deepcopy(base_lora),
      'pissa': deepcopy(base_pissa),
      'disha': deepcopy(base_disha),
      'alias': 'none',
    }

    raw = cls._to_plain_dict(raw_cfg)
    if not raw:
      return cfg

    alias = str(raw.get('type', 'none')).strip().lower()
    type_map = {
      'none': 'none',
      'lora': 'lora',
      'qlora': 'lora',
      'pissa': 'pissa',
      'disha': 'disha',
      'bone': 'disha',
      'bat': 'disha',
      'rslora': 'pissa',
      'rs-lora': 'pissa',
    }
    resolved_type = type_map.get(alias, 'none')
    cfg['type'] = resolved_type
    cfg['alias'] = alias or 'none'

    for key in ('freeze_non_peft', 'train_embeddings', 'train_head', 'train_layer_norms'):
      if key in raw:
        cfg[key] = bool(raw[key])

    if 'train_parts' in raw:
      parts = raw['train_parts']
      if isinstance(parts, (list, tuple)):
        cfg['train_parts'] = [str(p).strip() for p in parts if str(p).strip()] or cfg['train_parts']
      elif isinstance(parts, str) and parts.strip():
        cfg['train_parts'] = [parts.strip()]

    quant = str(raw.get('quantization', cfg['quantization'])).strip().lower()
    if resolved_type == 'lora' and alias == 'qlora' and not raw.get('quantization'):
      quant = 'nf4'
    if quant:
      cfg['quantization'] = quant

    lora_raw = cls._to_plain_dict(raw.get('lora'))
    if lora_raw:
      cfg['lora']['r'] = int(lora_raw.get('r', cfg['lora']['r']))
      cfg['lora']['alpha'] = float(lora_raw.get('alpha', cfg['lora']['alpha']))
      cfg['lora']['dropout'] = float(lora_raw.get('dropout', cfg['lora']['dropout']))
      if 'load_path' in lora_raw:
        cfg['lora']['load_path'] = str(lora_raw['load_path'])
      parts_val = lora_raw.get('target_parts') or lora_raw.get('parts')
      if parts_val:
        if isinstance(parts_val, (list, tuple)):
          cfg['lora']['target_parts'] = [str(p).strip().lower() for p in parts_val if str(p).strip()] or cfg['lora']['target_parts']
        elif isinstance(parts_val, str) and parts_val.strip():
          cfg['lora']['target_parts'] = [parts_val.strip().lower()]

    pissa_raw = cls._to_plain_dict(raw.get('pissa'))
    if pissa_raw:
      cfg['pissa']['r'] = int(pissa_raw.get('r', cfg['pissa']['r']))
      cfg['pissa']['svd_niter'] = int(pissa_raw.get('svd_niter', cfg['pissa']['svd_niter']))
      if 'load_path' in pissa_raw:
        cfg['pissa']['load_path'] = str(pissa_raw['load_path'])
      if 'init_path' in pissa_raw:
        cfg['pissa']['init_path'] = str(pissa_raw['init_path'])
      parts_val = pissa_raw.get('target_parts') or pissa_raw.get('parts')
      if parts_val:
        if isinstance(parts_val, (list, tuple)):
          cfg['pissa']['target_parts'] = [str(p).strip().lower() for p in parts_val if str(p).strip()] or cfg['pissa']['target_parts']
        elif isinstance(parts_val, str) and parts_val.strip():
          cfg['pissa']['target_parts'] = [parts_val.strip().lower()]

    disha_raw = cls._to_plain_dict(raw.get('disha'))
    if disha_raw:
      cfg['disha']['mode'] = str(disha_raw.get('mode', cfg['disha']['mode'])).strip().lower() or cfg['disha']['mode']
      cfg['disha']['r'] = int(disha_raw.get('r', cfg['disha']['r']))
      if 'load_path' in disha_raw:
        cfg['disha']['load_path'] = str(disha_raw['load_path'])
      parts_val = disha_raw.get('target_parts') or disha_raw.get('parts')
      if parts_val:
        if isinstance(parts_val, (list, tuple)):
          cfg['disha']['target_parts'] = [str(p).strip().lower() for p in parts_val if str(p).strip()] or cfg['disha']['target_parts']
        elif isinstance(parts_val, str) and parts_val.strip():
          cfg['disha']['target_parts'] = [parts_val.strip().lower()]

    if not cfg['lora']['target_parts']:
      cfg['lora']['target_parts'] = ['att', 'ffn']
    if not cfg['pissa']['target_parts']:
      cfg['pissa']['target_parts'] = ['att', 'ffn']
    if not cfg['disha']['target_parts']:
      cfg['disha']['target_parts'] = ['att', 'ffn']
    if cfg['disha']['mode'] not in {'bone', 'bat'}:
      cfg['disha']['mode'] = 'bone'

    return cfg

  @staticmethod
  def _prepare_peft_globals(peft_cfg: Dict[str, Any]) -> None:  # Configure global PEFT switches before constructing the model
    LORA_CONFIG['r'] = 0
    LORA_CONFIG['alpha'] = 0.0
    LORA_CONFIG['dropout'] = 0.0
    LORA_CONFIG['parts'] = set(peft_cfg['lora']['target_parts'])
    LORA_CONFIG['quant'] = peft_cfg['quantization'] != 'none'
    DiSHA_CONFIG['r'] = 0
    DiSHA_CONFIG['mode'] = peft_cfg['disha']['mode']
    DiSHA_CONFIG['parts'] = set(peft_cfg['disha']['target_parts'])

    peft_type = peft_cfg['type']
    if peft_type == 'lora':
      LORA_CONFIG['r'] = int(peft_cfg['lora']['r'])
      LORA_CONFIG['alpha'] = float(peft_cfg['lora']['alpha'])
      LORA_CONFIG['dropout'] = float(peft_cfg['lora']['dropout'])
      if LORA_CONFIG['r'] <= 0:
        raise ValueError('LoRA requires `r` > 0 when peft.type is "lora".')
    elif peft_type == 'pissa':
      LORA_CONFIG['r'] = int(peft_cfg['pissa']['r'])
      if LORA_CONFIG['r'] <= 0:
        raise ValueError('PiSSA requires `r` > 0 when peft.type is "pissa".')
    elif peft_type == 'disha':
      DiSHA_CONFIG['r'] = int(peft_cfg['disha']['r'])
      if DiSHA_CONFIG['r'] <= 0:
        raise ValueError('DiSHA requires `r` > 0 when peft.type is "disha".')

  def _apply_peft(self, args: TrainingArgs, peft_cfg: Dict[str, Any]) -> None:  # Freeze base weights, expose adapter params, and load checkpoints
    peft_type = peft_cfg['type']
    quant_type = peft_cfg['quantization']
    freeze = bool(peft_cfg.get('freeze_non_peft', True))
    if peft_type == 'none' and quant_type == 'none':
      return

    model = self.model
    if freeze:
      model.requires_grad_(False)

    if peft_cfg.get('train_embeddings', True) and hasattr(model, 'emb'):
      model.emb.weight.requires_grad = True
    if peft_cfg.get('train_head', True) and hasattr(model, 'head'):
      model.head.weight.requires_grad = True
    if peft_cfg.get('train_layer_norms', True):
      for name, param in model.named_parameters():
        if 'ln' in name.lower():
          param.requires_grad = True

    for part in peft_cfg.get('train_parts', []):
      lowered = str(part).lower()
      if not lowered:
        continue
      for name, param in model.named_parameters():
        if lowered in name.lower():
          param.requires_grad = True

    if peft_type in {'lora', 'pissa'}:
      for name, module in model.named_modules():
        for pname, param in module.named_parameters(recurse=False):
          if pname.startswith('lora_'):
            param.requires_grad = True
    if peft_type == 'disha':
      for name, module in model.named_modules():
        for pname, param in module.named_parameters(recurse=False):
          if 'disha' in pname:
            param.requires_grad = True

    if peft_type == 'pissa':
      svd_niter = int(peft_cfg['pissa']['svd_niter'])
      for name, module in model.named_modules():
        if hasattr(module, 'pissa_init') and callable(getattr(module, 'pissa_init')):
          module.pissa_init(svd_niter)

    load_path = ''
    if peft_type == 'lora':
      load_path = peft_cfg['lora'].get('load_path', '')
    elif peft_type == 'pissa':
      load_path = peft_cfg['pissa'].get('load_path', '')
    elif peft_type == 'disha':
      load_path = peft_cfg['disha'].get('load_path', '')
    load_path = str(load_path or '').strip()
    if load_path:
      state = torch.load(load_path, map_location='cpu')
      load_result = self.model.load_state_dict(state, strict=False)
      self._validate_load(load_result)

    if peft_type == 'pissa':
      init_path = str(peft_cfg['pissa'].get('init_path', '') or '').strip()
      if init_path and os.path.isfile(init_path):
        init_state = torch.load(init_path, map_location='cpu')
        for name, module in model.named_modules():
          if hasattr(module, 'pissa_load') and callable(getattr(module, 'pissa_load')):
            key_a = f'{name}.init_lora_A'
            key_b = f'{name}.init_lora_B'
            if key_a in init_state and key_b in init_state:
              module.pissa_load(init_state[key_a], init_state[key_b])

    if quant_type != 'none':
      if not torch.cuda.is_available():
        raise RuntimeError('Requested quantized training but CUDA is unavailable.')
      for module in model.modules():
        quant_fn = getattr(module, 'quant', None)
        if callable(quant_fn):
          quant_fn(quant_type)

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
    self.peft_config = self._normalise_peft_config(cfg.get('peft'))  # Canonicalise PEFT configuration for downstream use
    self._prepare_peft_globals(self.peft_config)  # Apply global adapter settings prior to model construction
    if 'checkpoint_path' not in cfg or not cfg.get('checkpoint_path'):  # Detect absent checkpoint configuration
      default_ckpt = Path(__file__).resolve().parents[3] / 'models' / 'blinkdl-rwkv7-g1a-1.5b' / 'rwkv-final.pth'  # Resolve default RWKV-7 checkpoint location bundled by the user
      if default_ckpt.is_file():  # Only adopt default when the checkpoint actually exists locally
        cfg['checkpoint_path'] = str(default_ckpt)  # Inject default checkpoint path so integration works out of the box
    self._configure_environment(max_seq_len, cfg)  # Seed environment variables so RWKV kernels compile with correct parameters
    args = self._build_args(vocab_size, d_model, n_layers, max_seq_len, cfg, self.peft_config)  # Generate TrainingArgs instance aligned with HRM configuration
    self.model = RWKV7(args)  # Instantiate RWKV-7 model using the assembled arguments
    self.peft_type = self.peft_config.get('type', 'none')  # Record active PEFT mode for diagnostics
    self.peft_alias = self.peft_config.get('alias', self.peft_type)  # Track provided PEFT alias (e.g. qlora -> lora)
    self.quantization = self.peft_config.get('quantization', 'none')  # Track requested quantisation mode
    if self.peft_type != 'none':
      adapter_r = self.peft_config.get(self.peft_type, {}).get('r', 'n/a')
      logger.info('RWKV7 PEFT enabled: type=%s alias=%s r=%s', self.peft_type, self.peft_alias, adapter_r)
    if self.quantization != 'none':
      logger.info('RWKV7 adapter quantization enabled: %s', self.quantization)
    self.kernel_backend = self._select_backend(args, cfg)  # Determine and activate the most efficient kernel backend before loading checkpoints
    checkpoint_path = Path(cfg.get('checkpoint_path', ''))  # Locate checkpoint path supplied by training config
    if not checkpoint_path.is_file():  # Validate that a checkpoint file is available before loading
      raise FileNotFoundError(f"RWKV-7 checkpoint missing at {checkpoint_path}")  # Provide informative error for missing weights
    state = torch.load(checkpoint_path, map_location='cpu')  # Load serialized parameter state onto CPU for deterministic initialization
    load_result = self.model.load_state_dict(state, strict=False)  # Populate RWKV-7 model weights while allowing head mismatches
    self._validate_load(load_result)  # Emit warnings when unexpected parameters are skipped or missing
    self.model.to(dtype=torch.bfloat16)  # Ensure base weights reside in bfloat16 to satisfy RWKV kernel expectations
    self._apply_peft(args, self.peft_config)  # Freeze / unfreeze parameters and load adapter checkpoints as necessary
    self.model.to(dtype=torch.bfloat16)  # Recast adapters / quantized parts to maintain kernel dtype contract
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
      r = r.to(torch.bfloat16)
      w = w.to(torch.bfloat16)
      k = k.to(torch.bfloat16)
      v = v.to(torch.bfloat16)
      a = a.to(torch.bfloat16)
      b = b.to(torch.bfloat16)
      B, T, HC = w.shape  # Unpack tensor dimensions
      if HC % HEAD_SIZE != 0:  # Validate divisibility between hidden channels and head size
        raise RuntimeError('hidden size is not divisible by RWKV head size for chunked kernel')  # Abort with descriptive error
      H = HC // HEAD_SIZE  # Compute number of heads dynamically
      shaped = [tensor.contiguous().view(B, T, H, HEAD_SIZE) for tensor in (r, w, k, v, a, b)]  # Reshape inputs to kernel layout
      outputs, _ = attn_chunked_cuda(*shaped)  # Execute forward pass and discard state output
      return outputs.view(B, T, HC)  # Restore original flattened layout

    rwkvop.RUN_CUDA_RWKV7g = run_chunked  # Patch RWKV operator to use wind chunked kernel
    self._patch_att_kernel(run_chunked)

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
      r = r.to(torch.bfloat16)
      w = w.to(torch.bfloat16)
      k = k.to(torch.bfloat16)
      v = v.to(torch.bfloat16)
      a = a.to(torch.bfloat16)
      b = b.to(torch.bfloat16)
      B, T, HC = w.shape  # Read tensor dimensions
      if HC % HEAD_SIZE != 0:  # Confirm divisibility
        raise RuntimeError('hidden size is not divisible by RWKV head size for longhead kernel')  # Provide error context
      H = HC // HEAD_SIZE  # Compute number of heads
      shaped = [tensor.contiguous().view(B, T, H, HEAD_SIZE) for tensor in (r, w, k, v, a, b)]  # Reshape into kernel expected format
      outputs, _ = attn_backstepping_longhead(*shaped)  # Execute kernel and ignore final state output
      return outputs.view(B, T, HC)  # Flatten back to original layout

    rwkvop.RUN_CUDA_RWKV7g = run_longhead  # Switch RWKV operator to wind longhead implementation
    self._patch_att_kernel(run_longhead)

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
      r = r.to(torch.bfloat16)
      w = w.to(torch.bfloat16)
      k = k.to(torch.bfloat16)
      v = v.to(torch.bfloat16)
      a = a.to(torch.bfloat16)
      b = b.to(torch.bfloat16)
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
    self._patch_att_kernel(run_fla)

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
    os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')  # Disable torch.compile for RWKV kernels to avoid dtype/assert issues
    os.environ.setdefault('DISABLE_TORCH_COMPILE', '1')  # Mirror disable flag recognized by some runtimes
    os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')  # Explicitly disable torch.compile decorators when possible
    os.environ.setdefault('PYTORCH_COMPILE_DISABLE', '1')  # PyTorch 2.4+ flag to bypass torch.compile graph capture

  @staticmethod
  def _build_args(
    vocab_size: int,
    d_model: int,
    n_layers: int,
    max_seq_len: int,
    cfg: dict,
    peft_cfg: Dict[str, Any],
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

    peft_type = peft_cfg.get('type', 'none')
    args.peft = peft_type
    args.quant = peft_cfg.get('quantization', 'none')
    args.train_parts = peft_cfg.get('train_parts', ['time', 'ln'])

    if peft_type == 'lora':
      lora_cfg = peft_cfg['lora']
      args.lora_config = {
        'lora_load': str(lora_cfg.get('load_path', '')),
        'lora_r': int(lora_cfg.get('r', 0)),
        'lora_alpha': float(lora_cfg.get('alpha', 0.0)),
        'lora_dropout': float(lora_cfg.get('dropout', 0.0)),
      }
    elif peft_type == 'pissa':
      pissa_cfg = peft_cfg['pissa']
      args.pissa_config = {
        'pissa_load': str(pissa_cfg.get('load_path', '')),
        'pissa_init': str(pissa_cfg.get('init_path', '')),
        'pissa_r': int(pissa_cfg.get('r', 0)),
        'svd_niter': int(pissa_cfg.get('svd_niter', 4)),
      }
    elif peft_type == 'disha':
      disha_cfg = peft_cfg['disha']
      args.disha_config = {
        'mode': str(disha_cfg.get('mode', 'bone')),
        'load': str(disha_cfg.get('load_path', '')),
        'r': int(disha_cfg.get('r', 0)),
      }
    else:
      args.peft = 'none'

    return args  # Provide fully populated TrainingArgs structure to the caller

  def _validate_load(self, load_result: object) -> None:
    missing = list(getattr(load_result, 'missing_keys', []))  # Extract missing keys from load_state_dict result for diagnostics
    unexpected = list(getattr(load_result, 'unexpected_keys', []))  # Extract unexpected keys reported during state loading

    def _is_tolerated(key: str) -> bool:
      if key == 'head.weight':
        return True
      if getattr(self, 'peft_type', 'none') != 'none' and ('lora_' in key or 'disha' in key):
        return True
      if key.endswith(('.v0', '.v1', '.v2')) or '.att.v' in key:
        return True
      return False

    critical_missing = [key for key in missing if not _is_tolerated(key)]
    critical_unexpected = [key for key in unexpected if not _is_tolerated(key)]
    if critical_missing or critical_unexpected:
      raise RuntimeError(f'RWKV-7 checkpoint mismatch: missing {critical_missing}, unexpected {critical_unexpected}')

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
