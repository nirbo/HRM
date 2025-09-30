import torch
from omegaconf import OmegaConf
import sys
import os
sys.path.append('src')
os.environ.setdefault('FUSED_KERNEL', '0')
os.environ.setdefault('RWKV_TRAIN_TYPE', 'none')
os.environ.setdefault('RWKV_MY_TESTING', 'x070')
os.environ.setdefault('WKV', 'cuda')

from hrm_lm.models.hybrid import HRMLanguageModel

cfg = OmegaConf.load('configs/default.yaml')
cfg.model.encoder.backend = 'rwkv7'
model = HRMLanguageModel(
  vocab_size=cfg.model.vocab_size,
  d_model=cfg.model.d_model,
  enc_layers=cfg.model.encoder.n_layers,
  dec_layers=cfg.model.decoder.n_layers,
  max_enc_len=cfg.model.encoder.max_seq_len,
  max_dec_len=cfg.model.decoder.max_seq_len,
  hrm_cfg=dict(cfg.model.hrm),
  bridge_cfg=dict(cfg.bridge),
  enc_backend='rwkv7',
  encoder_cfg=dict(cfg.model.encoder.get('encoder_cfg', {})),
)
model.eval()
with torch.no_grad():
  x = torch.randint(0, cfg.model.vocab_size, (1, 32))
  y_in = x
  logits = model(x, y_in)[0]
  print('logits shape', logits.shape)
