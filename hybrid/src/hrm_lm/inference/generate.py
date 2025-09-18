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
from hrm_lm.data.simple_tokenizer import BOS_ID, EOS_ID, SimpleTokenizer  # tokenizer utilities
from hrm_lm.data.synthetic import build_synthetic_dataset  # synthetic tokenizer builder

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # prefer GPU when available for generation

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

def load_artifact_tokenizer(checkpoint_dir: Optional[Path]):
  if checkpoint_dir is None:  # no checkpoint provided
    return None  # nothing to load
  tokenizer_path = checkpoint_dir / 'tokenizer.json'  # locate tokenizer artifact
  if not tokenizer_path.exists():  # artifact missing
    return None  # skip loading
  meta_path = checkpoint_dir / 'meta.json'  # locate metadata snapshot
  meta_data = None  # default metadata
  if meta_path.exists():  # metadata available
    with meta_path.open('r', encoding='utf-8') as handle:  # open metadata file
      meta_data = json.load(handle)  # parse metadata JSON
  return HFTokenizerAdapter(tokenizer_path, meta_data)  # return tokenizer adapter



@torch.no_grad()  # disable grad for inference
def generate(model: HRMLanguageModel, tokenizer, prompt: str, max_new_tokens: int = 64, device: str = 'cpu') -> str:
  model.eval()  # switch to eval mode
  dev = torch.device(device)  # normalize device handle
  enc = torch.tensor([tokenizer.encode(prompt, add_specials=True)], dtype=torch.long, device=dev)  # encode prompt
  bos = tokenizer.bos_id if hasattr(tokenizer, 'bos_id') else BOS_ID  # pick BOS id
  pad = tokenizer.pad_id if hasattr(tokenizer, 'pad_id') else 0  # pick PAD id
  eos = tokenizer.eos_id if hasattr(tokenizer, 'eos_id') else EOS_ID  # pick EOS id
  dec = torch.tensor([[bos]], dtype=torch.long, device=dev)  # seed decoder input
  enc_mask = (enc != pad).long()  # encoder mask
  for _ in range(max_new_tokens):  # iterate decoding steps
    dec_mask = (dec != pad).long()  # decoder mask
    out = model(enc, dec, enc_attn_mask=enc_mask, dec_attn_mask=dec_mask)  # forward pass
    logits = out['logits'][:, -1, :]  # get last-step logits
    next_token = torch.argmax(logits, dim=-1, keepdim=True)  # greedy choice
    dec = torch.cat([dec, next_token], dim=1)  # append token
    if int(next_token.item()) == eos:  # stop at EOS
      break  # exit loop
  return tokenizer.decode(dec[0].tolist())  # decode tokens


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
  text = generate(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens, device=device_str)  # run generation
  print(text)  # emit result


if __name__ == '__main__':
  main()  # invoke CLI
