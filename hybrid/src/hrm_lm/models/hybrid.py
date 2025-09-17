# Unified HRM+LM
import torch
import torch.nn as nn
from .encoder import LMEncoder
from .hrm_core import HRMCore
from .bridges import PromptToHRMBridge, HRMToPrefixBridge, HRMToCrossAttnBridge
from .decoder import LMDecoder

class HRMLanguageModel(nn.Module):
  def __init__(self, vocab_size, d_model, enc_layers, dec_layers, max_enc_len, max_dec_len, hrm_cfg, bridge_cfg, enc_backend='transformer'):
    super().__init__()
    self.encoder = LMEncoder(vocab_size, d_model, enc_layers, max_enc_len, backend=enc_backend)
    self.prompt2hrm = PromptToHRMBridge(d_model)
    self.hrm = HRMCore(d_model=d_model, h_len=hrm_cfg['h_len'], l_len=hrm_cfg['l_len'], h_layers=hrm_cfg['h_layers'], l_layers=hrm_cfg['l_layers'], h_cycles=hrm_cfg['h_cycles'], l_steps=hrm_cfg['l_steps'], approx_grad=hrm_cfg['approx_grad'], out_dim=hrm_cfg['out_dim'], use_halting=hrm_cfg.get('use_halting', False))
    self.bridge_type = bridge_cfg['type']
    if self.bridge_type == 'prefix':
      self.hrm2dec = HRMToPrefixBridge(d_model, bridge_cfg['prefix_len'])
    else:
      self.hrm2dec = HRMToCrossAttnBridge(d_model)
    self.decoder = LMDecoder(vocab_size, d_model, dec_layers, max_dec_len)

  def forward(self, input_ids, decoder_input_ids, enc_attn_mask=None, dec_attn_mask=None, labels=None):
    enc_h, cls = self.encoder(input_ids, enc_attn_mask)
    x = self.prompt2hrm(cls)
    z, _ = self.hrm(x, return_all=False)
    if self.bridge_type == 'prefix':
      mem = self.hrm2dec(z)
    else:
      mem = self.hrm2dec(z)
    logits = self.decoder(decoder_input_ids, mem, attention_mask=dec_attn_mask, memory_mask=None)
    loss = None
    if labels is not None:
      loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
    return {'logits': logits, 'loss': loss}
