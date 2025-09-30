"""Utility helpers for wiring pytorch_optimizer components into the trainer.

This module centralises optimizer, scheduler, and loss construction so the
training loop can stay lean while still offering rich configuration
capabilities.  All helpers accept plain dictionaries (or CLI-provided JSON
strings) and provide sensible defaults when optional parameters are omitted.
"""
from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
  ConstantLR,
  CosineAnnealingLR,
  CosineAnnealingWarmRestarts,
  CyclicLR,
  LambdaLR,
  MultiplicativeLR,
  MultiStepLR,
  OneCycleLR,
  StepLR,
)

from pytorch_optimizer import (
  create_optimizer,
  get_supported_loss_functions,
  get_supported_lr_schedulers,
  get_supported_optimizers,
  load_optimizer,
)
from pytorch_optimizer.loss import LOSS_FUNCTIONS
from pytorch_optimizer.lr_scheduler import CosineScheduler, LinearScheduler, PolyScheduler, REXScheduler
from pytorch_optimizer.lr_scheduler.chebyshev import get_chebyshev_schedule
from pytorch_optimizer.lr_scheduler.cosine_anealing import CosineAnnealingWarmupRestarts
from pytorch_optimizer.lr_scheduler.proportion import ProportionScheduler
from pytorch_optimizer.lr_scheduler.wsd import get_wsd_schedule

try:  # OmegaConf is optional at runtime.
  from omegaconf import DictConfig, OmegaConf
except ImportError:  # pragma: no cover - only executed if OmegaConf missing.
  DictConfig = None  # type: ignore[assignment]
  OmegaConf = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Public metadata -----------------------------------------------------------------

SUPPORTED_OPTIMIZERS: Tuple[str, ...] = tuple(sorted(get_supported_optimizers()))
SUPPORTED_LR_SCHEDULERS: Tuple[str, ...] = tuple(sorted(get_supported_lr_schedulers()))
SUPPORTED_LOSS_FUNCTIONS: Tuple[str, ...] = tuple(sorted(get_supported_loss_functions()))

OPTIMIZER_ALIASES: Dict[str, str] = {
  'adamw8bit': 'bnb_adamw8bit',
  'adamw_8bit': 'bnb_adamw8bit',
  'adam8bit': 'bnb_adam8bit',
  'adam_8bit': 'bnb_adam8bit',
  'sgd8bit': 'bnb_sgd8bit',
  'sgd_8bit': 'bnb_sgd8bit',
}

CAME_BETA_DEFAULT = 0.999
BETAS_REQUIRE_3 = {'came'}
_RWKV_UNSUPPORTED_OPTIMIZERS = {'alig', 'lomo', 'adalomo', 'adammini', 'muon', 'adamuon'}

LR_SCHEDULER_ALIASES: Dict[str, str] = {
  'cosine_restart': 'cosine_annealing_with_warm_restart',
  'cosine_warmup': 'cosine_annealing_with_warmup',
  'cosineannealing': 'cosine_annealing',
  'cosineannealingwarmrestarts': 'cosine_annealing_with_warm_restart',
  'cosineannealingwarmup': 'cosine_annealing_with_warmup',
  'onecycle': 'one_cycle',
}

LOSS_ALIASES: Dict[str, str] = {
  'ce': 'cross_entropy',
  'crossentropy': 'cross_entropy',
  'torch_cross_entropy': 'cross_entropy',
  'nll': 'cross_entropy',
}

# ---------------------------------------------------------------------------
# Helper utilities -------------------------------------------------------------


def _is_config_mapping(value: Any) -> bool:
  if isinstance(value, dict):
    return True
  if DictConfig is not None and isinstance(value, DictConfig):  # type: ignore[misc]
    return True
  return False


def _materialise_mapping(value: Any) -> Dict[str, Any]:
  if _is_config_mapping(value):
    if OmegaConf is not None and isinstance(value, DictConfig):  # type: ignore[misc]
      return dict(OmegaConf.to_container(value, resolve=True))
    return dict(value)
  return {}


def parse_kwargs(value: Union[None, str, Dict[str, Any]]) -> Dict[str, Any]:
  """Parse CLI/config overrides.

  Accepts dictionaries directly or JSON / Python literals supplied as strings.
  """
  if value is None:
    return {}
  if isinstance(value, dict):
    return dict(value)
  if DictConfig is not None and isinstance(value, DictConfig):  # type: ignore[misc]
    return dict(OmegaConf.to_container(value, resolve=True))
  if not isinstance(value, str):
    raise TypeError(f'Expected mapping or string for kwargs, received {type(value)!r}.')

  raw = value.strip()
  if not raw:
    return {}

  for parser in (json.loads, ast.literal_eval):
    try:
      parsed = parser(raw)
    except Exception:  # pragma: no cover - fallback path only
      continue
    if isinstance(parsed, dict):
      return dict(parsed)
  raise ValueError(f'Failed to parse kwargs string: {value!r}. Provide JSON or a Python dict literal.')


def normalise_name(name: Optional[str], *, default: str) -> str:
  if not name:
    return default
  resolved = name.strip().lower()
  if not resolved:
    return default
  return resolved


def _set_optimizer_lr(optimizer: Optimizer, lr: float) -> None:
  for group in optimizer.param_groups:
    scale = float(group.get('lr_scale', 1.0))
    group['lr'] = float(lr) * scale


def _should_decay(name: str, param: nn.Parameter, weight_decay: float, ban_list: Tuple[str, ...]) -> bool:
  if weight_decay <= 0.0:
    return False
  if param.dim() <= 1:
    return False
  lowered = name.lower()
  for pattern in ban_list:
    if pattern.lower() in lowered:
      return False
  return True


def _rwkv7_layerwise_groups(
  rwkv_module: nn.Module,
  *,
  weight_decay: float,
  ban_list: Tuple[str, ...],
  layerwise_lr: bool,
  pile_stage: int,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]]:
  lr_1x: List[nn.Parameter] = []
  lr_2x: List[nn.Parameter] = []
  lr_3x: List[nn.Parameter] = []
  decay: List[nn.Parameter] = []

  for name, param in rwkv_module.named_parameters():
    if not param.requires_grad:
      continue
    target: Optional[List[nn.Parameter]] = None
    if layerwise_lr and ("_w1" in name or "_w2" in name):
      target = lr_1x
    elif layerwise_lr and ("time_mix" in name or "time_maa" in name):
      target = lr_2x if pile_stage == 2 else lr_1x
    elif layerwise_lr and ("time_decay" in name or "time_daaaa" in name):
      target = lr_3x if pile_stage == 2 else lr_2x
    elif layerwise_lr and ("time_faaaa" in name):
      target = lr_2x if pile_stage == 2 else lr_1x
    elif layerwise_lr and ("time_first" in name):
      target = lr_3x
    elif _should_decay(name, param, weight_decay, ban_list):
      target = decay
    else:
      target = lr_1x
    target.append(param)

  return lr_1x, lr_2x, lr_3x, decay


def _build_rwkv7_optimizer(
  model: nn.Module,
  *,
  optimizer_name: str,
  base_lr: float,
  weight_decay: float,
  ban_list: Tuple[str, ...],
  effective_kwargs: Dict[str, Any],
  cfg_mapping: Dict[str, Any],
) -> Optional[Optimizer]:
  encoder = getattr(model, 'encoder', None)
  backend = getattr(encoder, 'backend', None)
  if backend != 'rwkv7':
    return None

  wrapper = getattr(encoder, 'enc', None)
  rwkv_module = getattr(wrapper, 'model', None)
  if rwkv_module is None:
    return None

  layerwise_lr = bool(cfg_mapping.get('layerwise_lr', True))
  pile_stage = int(cfg_mapping.get('rwkv_stage', 1))

  lr_1x, lr_2x, lr_3x, decay = _rwkv7_layerwise_groups(
    rwkv_module,
    weight_decay=weight_decay,
    ban_list=ban_list,
    layerwise_lr=layerwise_lr,
    pile_stage=pile_stage,
  )

  rwkv_param_ids = {id(p) for p in rwkv_module.parameters()}
  other_decay: List[nn.Parameter] = []
  other_main: List[nn.Parameter] = []
  for name, param in model.named_parameters():
    if id(param) in rwkv_param_ids or not param.requires_grad:
      continue
    if _should_decay(name, param, weight_decay, ban_list):
      other_decay.append(param)
    else:
      other_main.append(param)

  groups: List[Dict[str, Any]] = []

  def add_group(params: List[nn.Parameter], *, decay_value: float, scale: float) -> None:
    if params:
      groups.append({'params': params, 'weight_decay': decay_value, 'lr_scale': scale})

  if layerwise_lr:
    if pile_stage == 2:
      add_group(lr_1x, decay_value=0.0, scale=1.0)
      add_group(lr_2x, decay_value=0.0, scale=5.0)
      add_group(lr_3x, decay_value=0.0, scale=5.0)
    else:
      add_group(lr_1x, decay_value=0.0, scale=1.0)
      add_group(lr_2x, decay_value=0.0, scale=2.0)
      add_group(lr_3x, decay_value=0.0, scale=3.0)
  else:
    add_group(lr_1x, decay_value=0.0, scale=1.0)

  add_group(decay, decay_value=weight_decay, scale=1.0)
  add_group(other_main, decay_value=0.0, scale=1.0)
  add_group(other_decay, decay_value=weight_decay, scale=1.0)

  if not groups:
    return None

  if optimizer_name in _RWKV_UNSUPPORTED_OPTIMIZERS:
    raise ValueError(f"Optimizer '{optimizer_name}' is not supported with RWKV-7 custom grouping.")

  optimizer_class = load_optimizer(optimizer_name)
  optimizer = optimizer_class(groups, lr=float(base_lr), **effective_kwargs)
  return optimizer


# ---------------------------------------------------------------------------
# Optimizer construction ------------------------------------------------------


def build_optimizer(
  model: nn.Module,
  *,
  base_lr: float,
  cfg: Any,
  name_override: Optional[str] = None,
  cli_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Optimizer, str, Dict[str, Any]]:
  """Create an optimizer using pytorch_optimizer's ``create_optimizer`` helper.

  Returns the instantiated optimizer, the resolved optimizer name, and the
  merged keyword arguments used during construction.
  """
  cfg_mapping = _materialise_mapping(cfg)
  cfg_kwargs = parse_kwargs(cfg_mapping.get('kwargs'))

  # Backwards compatibility: recognise legacy scalar fields.
  if 'betas' in cfg_mapping and 'betas' not in cfg_kwargs:
    betas = cfg_mapping['betas']
    if isinstance(betas, Iterable):
      betas_tuple = tuple(float(b) for b in betas)
      if len(betas_tuple) == 2:
        cfg_kwargs['betas'] = betas_tuple
  if 'momentum' in cfg_mapping and 'momentum' not in cfg_kwargs:
    cfg_kwargs['momentum'] = float(cfg_mapping['momentum'])
  if 'eps' in cfg_mapping and 'eps' not in cfg_kwargs:
    cfg_kwargs['eps'] = float(cfg_mapping['eps'])

  effective_kwargs = {**cfg_kwargs, **(cli_kwargs or {})}

  raw_name = name_override or cfg_mapping.get('name') or 'adamw'
  resolved_name = normalise_name(raw_name, default='adamw')
  resolved_name = OPTIMIZER_ALIASES.get(resolved_name, resolved_name)

  if resolved_name not in SUPPORTED_OPTIMIZERS:
    raise ValueError(
      f"Unsupported optimizer '{raw_name}'. Available values: {', '.join(SUPPORTED_OPTIMIZERS)}"
    )

  betas_kw = effective_kwargs.get('betas')
  if betas_kw is not None:
    if isinstance(betas_kw, (list, tuple)):
      betas_tuple = tuple(float(b) for b in betas_kw)
    else:
      betas_tuple = tuple(float(b) for b in betas_kw)
    if resolved_name in BETAS_REQUIRE_3 and len(betas_tuple) == 2:
      betas_tuple = (betas_tuple[0], betas_tuple[1], CAME_BETA_DEFAULT)
    effective_kwargs['betas'] = betas_tuple

  weight_decay = float(cfg_mapping.get('weight_decay', 0.0))
  wd_ban_list = cfg_mapping.get('weight_decay_exclude', ('bias', 'LayerNorm.weight', 'LayerNorm.bias'))
  if isinstance(wd_ban_list, list):
    wd_ban_tuple = tuple(wd_ban_list)
  else:
    wd_ban_tuple = tuple(wd_ban_list)

  custom_kwargs = dict(effective_kwargs)
  custom_kwargs.pop('weight_decay', None)

  optimizer = _build_rwkv7_optimizer(
    model,
    optimizer_name=resolved_name,
    base_lr=float(base_lr),
    weight_decay=weight_decay,
    ban_list=wd_ban_tuple,
    effective_kwargs=custom_kwargs,
    cfg_mapping=cfg_mapping,
  )

  if optimizer is None:
    optimizer = create_optimizer(
      model,
      resolved_name,
      lr=float(base_lr),
      weight_decay=weight_decay,
      wd_ban_list=wd_ban_tuple,
      **effective_kwargs,
    )

  if resolved_name in BETAS_REQUIRE_3:
    for group in optimizer.param_groups:
      betas = group.get('betas')
      if isinstance(betas, tuple) and len(betas) == 2:
        group['betas'] = (betas[0], betas[1], CAME_BETA_DEFAULT)

  return optimizer, resolved_name, effective_kwargs


# ---------------------------------------------------------------------------
# Scheduler construction ------------------------------------------------------

@dataclass
class SchedulerSpec:
  scheduler: Optional[Any]
  handles_warmup: bool
  name: str
  kwargs: Dict[str, Any]


class SchedulerController:
  """Wraps a scheduler and manages optional linear warmup."""

  def __init__(
    self,
    optimizer: Optimizer,
    *,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    spec: SchedulerSpec,
  ) -> None:
    self.optimizer = optimizer
    self.base_lr = float(base_lr)
    self.total_steps = int(total_steps)
    self._scheduler = spec.scheduler
    self._manual_warmup = not spec.handles_warmup
    self._warmup_steps = max(0, int(warmup_steps)) if self._manual_warmup else 0
    self._step = 0
    self._last_lr = base_lr

  @property
  def scheduler(self) -> Optional[Any]:
    return self._scheduler

  @property
  def last_lr(self) -> float:
    return float(self._last_lr)

  def step(self) -> float:
    self._step += 1

    if self._manual_warmup and self._warmup_steps > 0 and self._step <= self._warmup_steps:
      ratio = self._step / float(self._warmup_steps)
      lr = self.base_lr * ratio
      _set_optimizer_lr(self.optimizer, lr)
      self._last_lr = lr
      return lr

    if self._scheduler is None:
      lr = self.base_lr
      _set_optimizer_lr(self.optimizer, lr)
      self._last_lr = lr
      return lr

    result = self._scheduler.step()
    if isinstance(result, list):
      lr = float(result[0])
    elif isinstance(result, (tuple, set)):
      lr = float(next(iter(result)))
    elif result is None:
      lr = float(self.optimizer.param_groups[0]['lr'])
    else:
      lr = float(result)

    self._last_lr = lr
    return lr

  def state_dict(self) -> Dict[str, Any]:
    state = {
      'step': self._step,
      'last_lr': self._last_lr,
      'manual_warmup': self._manual_warmup,
      'warmup_steps': self._warmup_steps,
      'base_lr': self.base_lr,
    }
    if self._scheduler is not None and hasattr(self._scheduler, 'state_dict'):
      state['scheduler'] = self._scheduler.state_dict()
    return state

  def load_state_dict(self, state: Dict[str, Any]) -> None:
    self._step = int(state.get('step', 0))
    self._last_lr = float(state.get('last_lr', self.base_lr))
    if self._manual_warmup:
      self._warmup_steps = int(state.get('warmup_steps', self._warmup_steps))
    if self._scheduler is not None and 'scheduler' in state:
      self._scheduler.load_state_dict(state['scheduler'])


def _build_constant_scheduler(
  optimizer: Optimizer,
  base_lr: float,
  total_steps: int,
  warmup_steps: int,
  kwargs: Dict[str, Any],
) -> SchedulerSpec:
  factor = float(kwargs.get('factor', 1.0))
  total_iters = int(kwargs.get('total_iters', max(0, total_steps - warmup_steps)))
  scheduler = ConstantLR(optimizer, factor=factor, total_iters=total_iters)
  return SchedulerSpec(scheduler=scheduler, handles_warmup=False, name='constant', kwargs=kwargs)


def _build_step_scheduler(
  optimizer: Optimizer,
  base_lr: float,
  total_steps: int,
  warmup_steps: int,
  kwargs: Dict[str, Any],
) -> SchedulerSpec:
  step_size = int(kwargs.get('step_size', max(1, (total_steps - warmup_steps) // 3)))
  gamma = float(kwargs.get('gamma', 0.5))
  scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
  return SchedulerSpec(scheduler=scheduler, handles_warmup=False, name='step', kwargs=kwargs)


def _build_multi_step_scheduler(
  optimizer: Optimizer,
  base_lr: float,
  total_steps: int,
  warmup_steps: int,
  kwargs: Dict[str, Any],
) -> SchedulerSpec:
  gross_steps = max(1, total_steps - warmup_steps)
  default_milestones = [gross_steps // 2, int(gross_steps * 0.75)]
  milestones = kwargs.get('milestones', default_milestones)
  if isinstance(milestones, str):
    milestones = [int(x) for x in milestones.split(',')]
  scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=float(kwargs.get('gamma', 0.1)))
  return SchedulerSpec(scheduler=scheduler, handles_warmup=False, name='multi_step', kwargs=kwargs)


def _build_multiplicative_scheduler(
  optimizer: Optimizer,
  base_lr: float,
  total_steps: int,
  warmup_steps: int,
  kwargs: Dict[str, Any],
) -> SchedulerSpec:
  factor = float(kwargs.get('factor', 0.95))
  scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda _: factor)
  return SchedulerSpec(scheduler=scheduler, handles_warmup=False, name='multiplicative', kwargs=kwargs)


def _build_one_cycle_scheduler(
  optimizer: Optimizer,
  base_lr: float,
  total_steps: int,
  warmup_steps: int,
  kwargs: Dict[str, Any],
) -> SchedulerSpec:
  scheduler = OneCycleLR(
    optimizer,
    max_lr=kwargs.get('max_lr', base_lr),
    total_steps=kwargs.get('total_steps', total_steps or 1),
    pct_start=float(kwargs.get('pct_start', 0.3)),
    anneal_strategy=kwargs.get('anneal_strategy', 'cos'),
    cycle_momentum=kwargs.get('cycle_momentum', False),
  )
  return SchedulerSpec(scheduler=scheduler, handles_warmup=False, name='one_cycle', kwargs=kwargs)


def _build_cyclic_scheduler(
  optimizer: Optimizer,
  base_lr: float,
  total_steps: int,
  warmup_steps: int,
  kwargs: Dict[str, Any],
) -> SchedulerSpec:
  base = float(kwargs.get('base_lr', base_lr / 25.0))
  max_lr = float(kwargs.get('max_lr', base_lr))
  mode = kwargs.get('mode', 'triangular2')
  scheduler = CyclicLR(
    optimizer,
    base_lr=base,
    max_lr=max_lr,
    step_size_up=int(kwargs.get('step_size_up', max(1, total_steps // 2))),
    mode=mode,
    cycle_momentum=False,
  )
  return SchedulerSpec(scheduler=scheduler, handles_warmup=False, name='cyclic', kwargs=kwargs)


def _build_cosine_scheduler(
  optimizer: Optimizer,
  base_lr: float,
  total_steps: int,
  warmup_steps: int,
  kwargs: Dict[str, Any],
) -> SchedulerSpec:
  t_max = int(kwargs.get('t_max', max(1, total_steps)))
  min_lr = float(kwargs.get('min_lr', 0.0))
  init_lr = float(kwargs.get('init_lr', min_lr))
  warmup = int(kwargs.get('warmup_steps', warmup_steps))
  scheduler = CosineScheduler(
    optimizer,
    t_max=t_max,
    max_lr=float(kwargs.get('max_lr', base_lr)),
    min_lr=min_lr,
    init_lr=init_lr,
    warmup_steps=max(0, warmup),
  )
  return SchedulerSpec(scheduler=scheduler, handles_warmup=True, name='cosine', kwargs=kwargs)


def _build_linear_scheduler(
  optimizer: Optimizer,
  base_lr: float,
  total_steps: int,
  warmup_steps: int,
  kwargs: Dict[str, Any],
) -> SchedulerSpec:
  scheduler = LinearScheduler(
    optimizer,
    t_max=int(kwargs.get('t_max', max(1, total_steps))),
    max_lr=float(kwargs.get('max_lr', base_lr)),
    min_lr=float(kwargs.get('min_lr', 0.0)),
    init_lr=float(kwargs.get('init_lr', 0.0)),
    warmup_steps=max(0, int(kwargs.get('warmup_steps', warmup_steps))),
  )
  return SchedulerSpec(scheduler=scheduler, handles_warmup=True, name='linear', kwargs=kwargs)


def _build_poly_scheduler(
  optimizer: Optimizer,
  base_lr: float,
  total_steps: int,
  warmup_steps: int,
  kwargs: Dict[str, Any],
) -> SchedulerSpec:
  scheduler = PolyScheduler(
    optimizer,
    t_max=int(kwargs.get('t_max', max(1, total_steps))),
    max_lr=float(kwargs.get('max_lr', base_lr)),
    min_lr=float(kwargs.get('min_lr', 0.0)),
    init_lr=float(kwargs.get('init_lr', 0.0)),
    warmup_steps=max(0, int(kwargs.get('warmup_steps', warmup_steps))),
    poly_order=float(kwargs.get('poly_order', 0.5)),
  )
  return SchedulerSpec(scheduler=scheduler, handles_warmup=True, name='poly', kwargs=kwargs)


def _build_cosine_annealing_scheduler(
  optimizer: Optimizer,
  base_lr: float,
  total_steps: int,
  warmup_steps: int,
  kwargs: Dict[str, Any],
) -> SchedulerSpec:
  t_max = int(kwargs.get('t_max', max(1, total_steps - warmup_steps)))
  scheduler = CosineAnnealingLR(
    optimizer,
    T_max=max(1, t_max),
    eta_min=float(kwargs.get('eta_min', 0.0)),
  )
  return SchedulerSpec(scheduler=scheduler, handles_warmup=False, name='cosine_annealing', kwargs=kwargs)


def _build_cosine_restart_scheduler(
  optimizer: Optimizer,
  base_lr: float,
  total_steps: int,
  warmup_steps: int,
  kwargs: Dict[str, Any],
) -> SchedulerSpec:
  first_cycle = int(kwargs.get('first_cycle_steps', max(1, total_steps // 4)))
  scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=first_cycle,
    T_mult=kwargs.get('cycle_mult', 1.0),
    eta_min=float(kwargs.get('eta_min', 0.0)),
  )
  return SchedulerSpec(scheduler=scheduler, handles_warmup=False, name='cosine_annealing_with_warm_restart', kwargs=kwargs)


def _build_cosine_warmup_restart_scheduler(
  optimizer: Optimizer,
  base_lr: float,
  total_steps: int,
  warmup_steps: int,
  kwargs: Dict[str, Any],
) -> SchedulerSpec:
  first_cycle = int(kwargs.get('first_cycle_steps', max(1, total_steps // 4)))
  scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=first_cycle,
    cycle_mult=float(kwargs.get('cycle_mult', 1.0)),
    max_lr=float(kwargs.get('max_lr', base_lr)),
    min_lr=float(kwargs.get('min_lr', 0.0)),
    warmup_steps=max(0, int(kwargs.get('warmup_steps', warmup_steps))),
    gamma=float(kwargs.get('gamma', 0.9)),
  )
  return SchedulerSpec(scheduler=scheduler, handles_warmup=True, name='cosine_annealing_with_warmup', kwargs=kwargs)


def _build_chebyshev_scheduler(
  optimizer: Optimizer,
  base_lr: float,
  total_steps: int,
  warmup_steps: int,
  kwargs: Dict[str, Any],
) -> SchedulerSpec:
  scheduler = get_chebyshev_schedule(
    optimizer,
    num_epochs=int(kwargs.get('num_epochs', max(1, total_steps))),
    is_warmup=bool(kwargs.get('is_warmup', False)),
  )
  return SchedulerSpec(scheduler=scheduler, handles_warmup=False, name='chebyshev', kwargs=kwargs)


def _build_rex_scheduler(
  optimizer: Optimizer,
  base_lr: float,
  total_steps: int,
  warmup_steps: int,
  kwargs: Dict[str, Any],
) -> SchedulerSpec:
  scheduler = REXScheduler(
    optimizer,
    total_steps=max(1, int(kwargs.get('total_steps', max(1, total_steps - warmup_steps)))),
    max_lr=float(kwargs.get('max_lr', base_lr)),
    min_lr=float(kwargs.get('min_lr', 0.0)),
  )
  return SchedulerSpec(scheduler=scheduler, handles_warmup=False, name='rex', kwargs=kwargs)


def _build_wsd_scheduler(
  optimizer: Optimizer,
  base_lr: float,
  total_steps: int,
  warmup_steps: int,
  kwargs: Dict[str, Any],
) -> SchedulerSpec:
  warmup = int(kwargs.get('num_warmup_steps', warmup_steps))
  stable = int(kwargs.get('num_stable_steps', max(1, (total_steps - warmup) // 2)))
  decay = int(kwargs.get('num_decay_steps', max(1, total_steps - warmup - stable)))
  scheduler = get_wsd_schedule(
    optimizer,
    num_warmup_steps=max(0, warmup),
    num_stable_steps=max(0, stable),
    num_decay_steps=max(1, decay),
    min_lr_ratio=float(kwargs.get('min_lr_ratio', 0.0)),
    num_cycles=float(kwargs.get('num_cycles', 0.5)),
    cooldown_type=kwargs.get('cooldown_type', '1-sqrt'),
  )
  return SchedulerSpec(scheduler=scheduler, handles_warmup=True, name='warmup_stable_decay', kwargs=kwargs)


def _build_proportion_scheduler(
  optimizer: Optimizer,
  base_lr: float,
  total_steps: int,
  warmup_steps: int,
  kwargs: Dict[str, Any],
) -> SchedulerSpec:
  base_scheduler_name = kwargs.get('base_scheduler', 'cosine')
  base_kwargs = parse_kwargs(kwargs.get('base_kwargs'))
  base_builder = SCHEDULER_BUILDERS.get(base_scheduler_name)
  if base_builder is None:
    raise ValueError(f"Proportion scheduler requires valid base_scheduler; received {base_scheduler_name!r}.")
  base_spec = base_builder(optimizer, base_lr, total_steps, warmup_steps, base_kwargs)
  max_value = float(kwargs.get('max_value', base_lr))
  min_value = float(kwargs.get('min_value', base_lr * 0.1))
  scheduler = ProportionScheduler(
    base_spec.scheduler,
    max_lr=max_value,
    min_lr=min_value,
    max_value=max_value,
    min_value=min_value,
  )
  return SchedulerSpec(scheduler=scheduler, handles_warmup=base_spec.handles_warmup, name='proportion', kwargs=kwargs)


SCHEDULER_BUILDERS: Dict[str, Callable[[Optimizer, float, int, int, Dict[str, Any]], SchedulerSpec]] = {
  'constant': _build_constant_scheduler,
  'step': _build_step_scheduler,
  'multi_step': _build_multi_step_scheduler,
  'multiplicative': _build_multiplicative_scheduler,
  'one_cycle': _build_one_cycle_scheduler,
  'cyclic': _build_cyclic_scheduler,
  'cosine': _build_cosine_scheduler,
  'linear': _build_linear_scheduler,
  'poly': _build_poly_scheduler,
  'cosine_annealing': _build_cosine_annealing_scheduler,
  'cosine_annealing_with_warm_restart': _build_cosine_restart_scheduler,
  'cosine_annealing_with_warmup': _build_cosine_warmup_restart_scheduler,
  'chebyshev': _build_chebyshev_scheduler,
  'rex': _build_rex_scheduler,
  'warmup_stable_decay': _build_wsd_scheduler,
  'proportion': _build_proportion_scheduler,
}


def build_scheduler_controller(
  optimizer: Optimizer,
  *,
  base_lr: float,
  total_steps: int,
  warmup_steps: int,
  cfg: Any,
  name_override: Optional[str] = None,
  cli_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[SchedulerController, SchedulerSpec]:
  cfg_mapping = _materialise_mapping(cfg)
  cfg_kwargs = parse_kwargs(cfg_mapping.get('kwargs'))
  effective_kwargs = {**cfg_kwargs, **(cli_kwargs or {})}

  raw_name = name_override or cfg_mapping.get('name') or 'cosine'
  resolved_name = normalise_name(raw_name, default='cosine')
  resolved_name = LR_SCHEDULER_ALIASES.get(resolved_name, resolved_name)

  if resolved_name == 'none':
    spec = SchedulerSpec(scheduler=None, handles_warmup=False, name='none', kwargs=effective_kwargs)
    return SchedulerController(
      optimizer,
      base_lr=base_lr,
      warmup_steps=warmup_steps,
      total_steps=total_steps,
      spec=spec,
    ), spec

  if resolved_name not in SCHEDULER_BUILDERS:
    raise ValueError(
      f"Unsupported lr scheduler '{raw_name}'. Available values: {', '.join(sorted(SCHEDULER_BUILDERS.keys()))}"
    )

  builder = SCHEDULER_BUILDERS[resolved_name]
  spec = builder(optimizer, float(base_lr), int(total_steps), int(warmup_steps), effective_kwargs)
  controller = SchedulerController(
    optimizer,
    base_lr=base_lr,
    warmup_steps=warmup_steps,
    total_steps=total_steps,
    spec=spec,
  )
  return controller, spec


# ---------------------------------------------------------------------------
# Loss construction ----------------------------------------------------------

class LossAdapter:
  """Wrap loss modules to handle token masking uniformly."""

  def __init__(self, loss_module: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], *, ignore_index: int = -100) -> None:
    self.loss_module = loss_module
    self.ignore_index = ignore_index

  def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    vocab_dim = logits.size(-1)
    logits_flat = logits.reshape(-1, vocab_dim)
    labels_flat = labels.reshape(-1)
    if self.ignore_index is not None:
      mask = labels_flat != self.ignore_index
      if mask.ndim > 0 and mask.sum() == 0:
        zero = logits_flat.sum() * 0.0
        zero.requires_grad = True
        return zero
      logits_filtered = logits_flat[mask]
      labels_filtered = labels_flat[mask]
    else:
      logits_filtered = logits_flat
      labels_filtered = labels_flat
    return self.loss_module(logits_filtered, labels_filtered)


def build_loss(
  *,
  loss_name: Optional[str],
  cfg: Any,
  kwargs_override: Optional[Dict[str, Any]] = None,
  ignore_index: int = -100,
) -> Tuple[LossAdapter, str, Dict[str, Any]]:
  cfg_mapping = _materialise_mapping(cfg)
  cfg_kwargs = parse_kwargs(cfg_mapping.get('kwargs'))
  effective_kwargs = {**cfg_kwargs, **(kwargs_override or {})}

  raw_name = loss_name or cfg_mapping.get('name') or 'cross_entropy'
  resolved_name = normalise_name(raw_name, default='cross_entropy')
  resolved_name = LOSS_ALIASES.get(resolved_name, resolved_name)

  if resolved_name in ('cross_entropy', 'torch_cross_entropy'):
    label_smoothing = float(effective_kwargs.get('label_smoothing', cfg_mapping.get('label_smoothing', 0.0)))
    reduction = effective_kwargs.get('reduction', cfg_mapping.get('reduction', 'mean'))
    ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=label_smoothing, reduction=reduction)
    return LossAdapter(ce_loss, ignore_index=ignore_index), 'cross_entropy', effective_kwargs

  if resolved_name not in SUPPORTED_LOSS_FUNCTIONS:
    raise ValueError(
      f"Unsupported loss '{raw_name}'. Available values: cross_entropy, "
      f"{', '.join(SUPPORTED_LOSS_FUNCTIONS)}"
    )

  loss_cls = LOSS_FUNCTIONS[resolved_name]
  loss_obj = loss_cls(**effective_kwargs)
  # Many pytorch_optimizer losses expect logits; rely on caller to provide sensible tensors.
  def _loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return loss_obj(logits, labels)

  return LossAdapter(_loss_fn, ignore_index=ignore_index), resolved_name, effective_kwargs


__all__ = [
  'build_optimizer',
  'build_scheduler_controller',
  'build_loss',
  'LossAdapter',
  'SchedulerController',
  'SchedulerSpec',
  'SUPPORTED_OPTIMIZERS',
  'SUPPORTED_LR_SCHEDULERS',
  'SUPPORTED_LOSS_FUNCTIONS',
  'parse_kwargs',
]
