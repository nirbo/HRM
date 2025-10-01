"""Greedy generation helper and CLI for HRM-LM."""  # module summary

import argparse  # CLI parsing
import json  # parse checkpoint metadata
import os  # filesystem helpers
from pathlib import Path  # filesystem path utilities
from typing import Optional  # optional type hints

import torch  # tensor ops
from omegaconf import OmegaConf  # config loading
from tokenizers import Tokenizer  # load Hugging Face tokenizer files

from hrm_lm.models.hybrid import HRMLanguageModel  # model type hints
from hrm_lm.training.train import make_model  # model builder
from hrm_lm.tokenizers import RWKVTokenizer, load_rwkv_vocab  # RWKV vocabulary utilities
from hrm_lm.data.simple_tokenizer import BOS_ID, EOS_ID, SimpleTokenizer  # tokenizer utilities
from hrm_lm.data.synthetic import build_synthetic_dataset  # synthetic tokenizer builder
from chat_template import render_template  # chat template helpers

if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
  os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # prefer GPU when available for generation


def apply_chat_template(prompt: str, template: Optional[str], system: Optional[str]) -> str:
  """Format ``prompt`` using the requested chat template.

  Args:
    prompt: Raw user prompt text.
    template: Either ``None``, the literal string ``'canonical'``, or a path to a
      template file.
    system: Optional system instruction to include when rendering canonical
      templates.

  Returns:
    A formatted prompt string suitable for tokenisation.
  """
  if not template:
    return prompt

  if template.lower() == 'canonical':
    messages = []
    if system:
      messages.append({'role': 'system', 'content': system})
    messages.append({'role': 'user', 'content': prompt})
    rendered = render_template({'messages': messages}, api='openai-chat')
    # Open-ended assistant block so generation appends the reply.
    return f"{rendered}\n\n<<ASSISTANT>>\n"

  template_path = Path(template)
  if not template_path.exists():
    raise FileNotFoundError(f"chat template {template} not found")
  text = template_path.read_text(encoding='utf-8')
  if '{prompt}' in text:
    return text.replace('{prompt}', prompt)
  # Append user prompt to template verbatim when no placeholder is present.
  return f"{text}\n{prompt}"

def resolve_device(requested: Optional[str]) -> torch.device:
  desired = requested or DEFAULT_DEVICE  # choose requested device or default preference
  device = torch.device(desired)  # convert string descriptor to torch device
  if device.type == 'cuda' and not torch.cuda.is_available():  # guard against unavailable CUDA
    raise RuntimeError('CUDA requested but not available')  # fail fast with informative error
  return device  # return resolved device handle

def load_config_from_sources(config_path: Optional[str], default_path: str, checkpoint_state: Optional[dict]):
  if config_path is not None:  # prefer explicit CLI config
    return OmegaConf.load(config_path)  # load user-specified YAML
  if isinstance(checkpoint_state, dict) and 'cfg' in checkpoint_state:  # check checkpoint payload
    return OmegaConf.create(checkpoint_state['cfg'])  # reconstruct config from checkpoint snapshot
  return OmegaConf.load(default_path)  # fall back to repo default config

class HFTokenizerAdapter:
  def __init__(self, tokenizer_path: Path, meta: Optional[dict]):
    self._tokenizer = Tokenizer.from_file(str(tokenizer_path))  # load serialized tokenizer
    pad_meta = (meta or {}).get('pad_id')  # fetch pad id from metadata when present
    bos_meta = (meta or {}).get('bos_id')  # fetch bos id from metadata when present
    eos_meta = (meta or {}).get('eos_id')  # fetch eos id from metadata when present
    self.pad_id = pad_meta if pad_meta is not None else self._tokenizer.token_to_id('<pad>')  # resolve pad id
    self.bos_id = bos_meta if bos_meta is not None else self._tokenizer.token_to_id('<bos>')  # resolve bos id
    self.eos_id = eos_meta if eos_meta is not None else self._tokenizer.token_to_id('<eos>')  # resolve eos id
    for fallback_name, attr_name in (('<pad>', 'pad_id'), ('<bos>', 'bos_id'), ('<eos>', 'eos_id')):  # iterate special tokens
      if getattr(self, attr_name) is None:  # check unresolved token
        setattr(self, attr_name, self._tokenizer.token_to_id(fallback_name))  # attempt fallback lookup
  def encode(self, text: str, add_specials: bool = True):
    ids = self._tokenizer.encode(text).ids  # convert text to token ids
    if not add_specials:  # no special token handling
      return ids  # return raw ids
    sequence = []  # accumulate sequence with specials
    if self.bos_id is not None:  # prepend BOS when available
      sequence.append(self.bos_id)  # add BOS id
    sequence.extend(ids)  # append encoded ids
    if self.eos_id is not None:  # append EOS when available
      sequence.append(self.eos_id)  # add EOS id
    return sequence  # return finalized sequence
  def decode(self, ids, skip_specials: bool = True):
    specials = {token_id for token_id in (self.pad_id, self.bos_id, self.eos_id) if token_id is not None}  # collect special ids
    filtered = [idx for idx in ids if not (skip_specials and idx in specials)]  # optionally filter specials
    return self._tokenizer.decode(filtered)  # decode ids back to text


class RWKVTokenizerAdapter:
  def __init__(self, tokenizer_path: Path, meta: Optional[dict]):
    vocab_items = load_rwkv_vocab(tokenizer_path)  # load RWKV vocabulary entries from disk
    self._tokenizer = RWKVTokenizer(vocab_items)  # instantiate RWKV tokenizer with the loaded entries
    self._id_to_bytes = {token_id: token_bytes for token_id, token_bytes in vocab_items}  # cache id-to-bytes mapping for decoding
    metadata = meta or {}  # normalize metadata dictionary for special-token overrides
    pad_meta = metadata.get('pad_id')  # extract optional pad id override
    bos_meta = metadata.get('bos_id')  # extract optional BOS id override
    eos_meta = metadata.get('eos_id')  # extract optional EOS id override
    try:
      self.pad_id = int(pad_meta)  # coerce pad id override to integer when provided
    except (TypeError, ValueError):
      self.pad_id = self._tokenizer.pad_token_id  # fall back to tokenizer default pad id
    try:
      self.bos_id = int(bos_meta)  # coerce BOS id override to integer when provided
    except (TypeError, ValueError):
      self.bos_id = self._tokenizer.bos_token_id  # fall back to tokenizer default BOS id
    try:
      self.eos_id = int(eos_meta)  # coerce EOS id override to integer when provided
    except (TypeError, ValueError):
      self.eos_id = self._tokenizer.eos_token_id  # fall back to tokenizer default EOS id

  def encode(self, text: str, add_specials: bool = True):
    ids = self._tokenizer.encode(text)  # tokenize text into RWKV token ids
    if not add_specials:  # respect add_specials flag
      return ids  # return raw token sequence
    sequence = []  # assemble sequence with special tokens
    if self.bos_id is not None:  # optionally prepend BOS token
      sequence.append(self.bos_id)  # add BOS id to sequence
    sequence.extend(ids)  # append base token ids
    if self.eos_id is not None:  # optionally append EOS token
      sequence.append(self.eos_id)  # add EOS id to sequence
    return sequence  # return finalized token id sequence

  def decode(self, ids, skip_specials: bool = True):
    specials = {token_id for token_id in (self.pad_id, self.bos_id, self.eos_id) if token_id is not None}  # collect special ids for filtering
    buffer = bytearray()  # accumulate decoded byte stream
    for raw_id in ids:  # iterate requested token ids
      token_id = int(raw_id)  # normalize token id to python int
      if skip_specials and token_id in specials:  # optionally skip special tokens
        continue  # ignore this token id
      token_bytes = self._id_to_bytes.get(token_id)  # fetch raw bytes for token id
      if token_bytes is None:  # guard against missing vocabulary entries
        continue  # skip unknown ids
      buffer.extend(token_bytes)  # append token bytes to output buffer
    return buffer.decode('utf-8', errors='replace')  # decode accumulated bytes into utf-8 string


def _build_tokenizer_adapter(candidate_path: Path, meta: Optional[dict]):
  if not candidate_path.exists():  # ensure candidate file is present
    raise FileNotFoundError(f'tokenizer artifact {candidate_path} not found')  # surface missing artifacts explicitly
  hf_error: Optional[Exception] = None  # track huggingface loader failure for diagnostics
  try:
    return HFTokenizerAdapter(candidate_path, meta)  # prefer huggingface tokenizer loader when compatible
  except Exception as exc:  # capture huggingface loader errors for reuse
    hf_error = exc  # remember huggingface loader exception for later reporting
  try:
    return RWKVTokenizerAdapter(candidate_path, meta)  # attempt RWKV vocabulary loading next
  except Exception as exc:  # capture RWKV loader errors
    message = f'Failed to load tokenizer from {candidate_path}: HF error={hf_error}; RWKV error={exc}'  # compose aggregated diagnostic message
    raise RuntimeError(message) from exc  # raise combined error with context for troubleshooting


def load_artifact_tokenizer(checkpoint_dir: Optional[Path]):
  if checkpoint_dir is None:  # no checkpoint provided
    return None  # nothing to load
  meta_data = None  # placeholder for parsed metadata
  meta_path = checkpoint_dir / 'meta.json'  # locate metadata snapshot
  if meta_path.exists():  # metadata available
    with meta_path.open('r', encoding='utf-8') as handle:  # open metadata file
      try:
        meta_data = json.load(handle)  # parse metadata JSON payload
      except json.JSONDecodeError as exc:  # handle malformed metadata
        raise RuntimeError(f'Failed to parse {meta_path}: {exc}') from exc  # surface metadata parsing errors
  candidates = []  # collect candidate tokenizer files to inspect
  for name in ('tokenizer.json', 'tokenizer_rwkv7_vocab.json', 'tokenizer_rwkv_vocab.json'):  # enumerate common artifact filenames
    candidates.append((checkpoint_dir / name, False))  # append optional candidate path
  if isinstance(meta_data, dict):  # honor tokenizer pointer from metadata when available
    token_file = meta_data.get('tokenizer_file')  # extract tokenizer file hint
    if token_file:  # ensure hint is not empty
      meta_candidate = Path(str(token_file))  # normalize hint to Path
      if not meta_candidate.is_absolute():  # resolve relative paths w.r.t. checkpoint directory
        meta_candidate = (checkpoint_dir / meta_candidate).resolve()  # convert relative path to absolute
      candidates.append((meta_candidate, True))  # treat metadata-specified tokenizer as required
  seen = set()  # avoid reprocessing duplicate paths
  errors = []  # collect loading errors for diagnostics
  for candidate_path, required in candidates:  # iterate candidate tokenizer paths
    resolved = Path(candidate_path).resolve()  # normalize candidate path to absolute form
    if resolved in seen:  # skip duplicates
      continue  # move to next candidate
    seen.add(resolved)  # mark path as visited
    if not resolved.exists():  # handle missing files
      if required:  # metadata-promised tokenizer missing
        errors.append(f'{resolved}: file not found')  # record missing required artifact
      continue  # examine next candidate
    try:
      return _build_tokenizer_adapter(resolved, meta_data)  # attempt to construct tokenizer adapter from artifact
    except Exception as exc:  # record loading failures
      errors.append(f'{resolved}: {exc}')  # store failure details for aggregated error
      continue  # continue evaluating other candidates
  if errors:  # check if any candidate attempts failed
    joined = '; '.join(errors)  # join error messages for readability
    raise RuntimeError(f'Failed to load tokenizer artifacts: {joined}')  # surface aggregated failure
  return None  # fall back to synthetic/simple tokenizers when no artifacts available



def _sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
  if temperature <= 0:
    temperature = 1e-5  # prevent divide-by-zero
  logits = logits / temperature
  logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)  # stabilise logits before softmax
  probs = torch.softmax(logits, dim=-1)
  probs = torch.nan_to_num(probs, nan=0.0)  # guard against residual NaNs from numerical issues
  if top_p < 1.0:
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative > top_p
    mask[..., 0] = False
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    next_token = torch.multinomial(sorted_probs, num_samples=1)
    next_token = sorted_indices.gather(-1, next_token)
  else:
    next_token = torch.multinomial(probs, num_samples=1)
  return next_token


@torch.no_grad()  # disable grad for inference
def generate(model: HRMLanguageModel, tokenizer, prompt: str, max_new_tokens: int = 64, device: str = 'cpu', temperature: float = 1.0, top_p: float = 1.0) -> str:
  model.eval()  # switch to eval mode
  dev = torch.device(device)  # normalize device handle
  prompt_ids = tokenizer.encode(prompt, add_specials=True)  # encode prompt once (with specials)
  enc = torch.tensor([prompt_ids], dtype=torch.long, device=dev)  # encoder input uses full prompt
  bos = tokenizer.bos_id if hasattr(tokenizer, 'bos_id') else BOS_ID  # pick BOS id
  pad = tokenizer.pad_id if hasattr(tokenizer, 'pad_id') else 0  # pick PAD id
  eos = tokenizer.eos_id if hasattr(tokenizer, 'eos_id') else EOS_ID  # pick EOS id
  if prompt_ids and prompt_ids[-1] == eos:  # remove trailing EOS to avoid duplicate
    dec_seed = prompt_ids[:-1]
  else:
    dec_seed = prompt_ids
  if not dec_seed:  # ensure decoder has at least BOS when prompt was empty
    dec_seed = [bos] if bos is not None else []
  dec = torch.tensor([dec_seed], dtype=torch.long, device=dev)  # seed decoder with prompt tokens
  base_len = dec.size(1)  # remember prompt length for continuation slicing
  enc_mask = (enc != pad).bool()  # encoder mask as boolean mask
  for _ in range(max_new_tokens):  # iterate decoding steps
    dec_mask = (dec != pad).bool()  # decoder mask as boolean mask
    out = model(enc, dec, enc_attn_mask=enc_mask, dec_attn_mask=dec_mask)  # forward pass
    logits = out['logits'][:, -1, :]  # get last-step logits
    next_token = _sample_next_token(logits, temperature=temperature, top_p=top_p)  # stochastic choice
    dec = torch.cat([dec, next_token], dim=1)  # append token
    if int(next_token.item()) == eos:  # stop at EOS
      break  # exit loop
  full_tokens = dec[0].tolist()  # collect full sequence (prompt + continuation)
  continuation_ids = full_tokens[base_len:]  # isolate generated continuation
  continuation_text = tokenizer.decode(continuation_ids, skip_specials=True)  # decode continuation only
  return prompt + continuation_text  # append continuation to original prompt


def _default_config_path() -> str:
  base = os.path.dirname(os.path.dirname(__file__))  # locate src root
  return os.path.join(base, 'configs', 'default.yaml')  # build default config path


def main() -> None:
  ap = argparse.ArgumentParser(description='Generate text with HRM-LM')  # CLI description
  ap.add_argument('--config', default=None, help='Path to model config YAML')  # config path
  ap.add_argument('--checkpoint', default=None, help='Optional checkpoint to load')  # checkpoint path
  ap.add_argument('--prompt', required=True, help='Prompt to condition on')  # prompt text
  ap.add_argument('--device', default=DEFAULT_DEVICE, help='Torch device for inference (defaults to GPU when available)')  # device selection
  ap.add_argument('--dataset', default='synthetic', help='Tokenizer source hint (synthetic|auto|none)')  # tokenizer source
  ap.add_argument('--max_new_tokens', type=int, default=64, help='Maximum tokens to decode')  # generation length
  ap.add_argument('--temperature', type=float, default=1.0, help='Softmax temperature for sampling (default: 1.0)')
  ap.add_argument('--top_p', type=float, default=1.0, help='Top-p nucleus sampling threshold (default: 1.0 = disabled)')
  ap.add_argument('--chat_template', default=None, help="Optional chat template ('canonical' or path to a template file).")
  ap.add_argument('--system', default=None, help='Optional system instruction when using --chat_template canonical.')
  args = ap.parse_args()  # parse CLI args

  dataset_hint = (args.dataset or 'auto').lower()  # normalize dataset hint
  checkpoint_path = Path(args.checkpoint) if args.checkpoint else None  # normalize checkpoint path
  checkpoint_dir = checkpoint_path.parent if checkpoint_path is not None else None  # derive checkpoint directory
  device = resolve_device(args.device)  # resolve desired device with GPU preference
  checkpoint_state = None  # placeholder for checkpoint payload
  if checkpoint_path is not None:  # checkpoint provided
    if not checkpoint_path.exists():  # validate path
      raise FileNotFoundError(f'Checkpoint {checkpoint_path} not found')  # fail fast on missing checkpoint
    checkpoint_state = torch.load(checkpoint_path, map_location=device)  # load checkpoint with resolved device

  cfg = load_config_from_sources(args.config, _default_config_path(), checkpoint_state)  # resolve configuration source
  model = make_model(cfg).to(device)  # instantiate model on target device
  if checkpoint_state is not None:  # load weights when checkpoint supplied
    state_dict = checkpoint_state.get('state_dict') if isinstance(checkpoint_state, dict) else checkpoint_state  # extract raw state dict
    model.load_state_dict(state_dict)  # populate model parameters

  tokenizer = None  # initialize tokenizer handle
  artifact_tokenizer = load_artifact_tokenizer(checkpoint_dir)  # attempt to load tokenizer artifact
  if artifact_tokenizer is not None:  # artifact available
    tokenizer = artifact_tokenizer  # use artifact tokenizer
  elif dataset_hint == 'synthetic':  # preserve synthetic fallback
    tokenizer, _ = build_synthetic_dataset(n=2000, seed=cfg.train.seed)  # build synthetic tokenizer
  else:  # final fallback
    tokenizer = SimpleTokenizer()  # use simple whitespace tokenizer
    tokenizer.fit([args.prompt])  # build minimal vocabulary from prompt

  device_str = str(device)  # stringify device for generator helper
  formatted_prompt = apply_chat_template(args.prompt, args.chat_template, args.system)

  text = generate(
    model,
    tokenizer,
    formatted_prompt,
    max_new_tokens=args.max_new_tokens,
    device=device_str,
    temperature=args.temperature,
    top_p=args.top_p,
  )  # run generation
  print(text)  # emit result


if __name__ == '__main__':
  main()  # invoke CLI
