"""Greedy generation helper and CLI for HRM-LM."""  # module summary

import os  # filesystem helpers
import argparse  # CLI parsing

import torch  # tensor ops
from omegaconf import OmegaConf  # config loading

from hrm_lm.models.hybrid import HRMLanguageModel  # model type hints
from hrm_lm.training.train import make_model  # model builder
from hrm_lm.data.simple_tokenizer import BOS_ID, EOS_ID, SimpleTokenizer  # tokenizer utilities
from hrm_lm.data.synthetic import build_synthetic_dataset  # synthetic tokenizer builder


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
  ap.add_argument('--device', default='cpu', help='Torch device for inference')  # device selection
  ap.add_argument('--dataset', default='synthetic', help='Tokenizer source (synthetic|none)')  # tokenizer source
  ap.add_argument('--max_new_tokens', type=int, default=64, help='Maximum tokens to decode')  # generation length
  args = ap.parse_args()  # parse CLI args

  cfg_path = args.config or _default_config_path()  # resolve config path
  cfg = OmegaConf.load(cfg_path)  # load config
  device = torch.device(args.device)  # resolve device
  model = make_model(cfg).to(device)  # instantiate model
  if args.checkpoint:  # load checkpoint if provided
    state = torch.load(args.checkpoint, map_location=device)  # load state dict
    if isinstance(state, dict) and 'state_dict' in state:  # check wrapper
      model.load_state_dict(state['state_dict'])  # load nested state
    else:
      model.load_state_dict(state)  # load plain state

  if args.dataset == 'synthetic':  # select tokenizer source
    tokenizer, _ = build_synthetic_dataset(n=2000, seed=cfg.train.seed)  # build tokenizer from synthetic data
  else:
    tokenizer = SimpleTokenizer()  # fallback tokenizer
    tokenizer.fit([args.prompt])  # prime vocab with prompt

  text = generate(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens, device=args.device)  # run generation
  print(text)  # emit result


if __name__ == '__main__':
  main()  # invoke CLI
