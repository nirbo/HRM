# Unified HRM+LM
import torch
import torch.nn as nn
from .encoder import LMEncoder
from .hrm_core import HRMCore
from .bridges import PromptToHRMBridge, HRMToPrefixBridge, HRMToCrossAttnBridge, HRMGate  # include gating
from .decoder import LMDecoder

class HRMLanguageModel(nn.Module):
  def __init__(self, vocab_size, d_model, enc_layers, dec_layers, max_enc_len, max_dec_len, hrm_cfg, bridge_cfg, enc_backend='transformer'):
    super().__init__()
    self.encoder = LMEncoder(vocab_size, d_model, enc_layers, max_enc_len, backend=enc_backend)
    self.prompt2hrm = PromptToHRMBridge(d_model)
    use_halting = hrm_cfg.get('use_halting', False)  # cache halting flag
    self.hrm = HRMCore(d_model=d_model, h_len=hrm_cfg['h_len'], l_len=hrm_cfg['l_len'], h_layers=hrm_cfg['h_layers'], l_layers=hrm_cfg['l_layers'], h_cycles=hrm_cfg['h_cycles'], l_steps=hrm_cfg['l_steps'], approx_grad=hrm_cfg['approx_grad'], out_dim=hrm_cfg['out_dim'], use_halting=use_halting)  # build HRM core
    self.use_halting = use_halting  # store halting enablement
    self.halting_weight = hrm_cfg.get('halting_weight', 0.0)  # store halting loss weight
    default_target = 1.0 if use_halting else float(hrm_cfg.get('h_cycles', 0))  # prefer attainable halting target when enabled
    self.halting_target = float(hrm_cfg.get('halting_target', default_target))  # resolve halting target scalar
    self.deep_supervision = hrm_cfg.get('deep_supervision', False)  # toggle deep supervision
    self.ds_weight = hrm_cfg.get('ds_weight', 0.0)  # weight for deep supervision loss
    self.bridge_type = bridge_cfg['type']
    if self.bridge_type == 'prefix':
      self.hrm2dec = HRMToPrefixBridge(d_model, bridge_cfg['prefix_len'])
    else:
      self.hrm2dec = HRMToCrossAttnBridge(d_model)
    self.decoder = LMDecoder(vocab_size, d_model, dec_layers, max_dec_len)
    self.hrm_gate = HRMGate(d_model)  # learnable bridge gate
    self.register_buffer('gate_scale', torch.tensor(1.0, dtype=torch.float32))  # Scalar multiplier controlling HRM gate strength.

  def forward(self, input_ids, decoder_input_ids, enc_attn_mask=None, dec_attn_mask=None, labels=None):
    enc_h, cls = self.encoder(input_ids, enc_attn_mask)
    x = self.prompt2hrm(cls)
    need_aux = self.halting_weight > 0 or self.deep_supervision  # determine if auxiliary outputs required
    z, aux = self.hrm(x, return_all=need_aux)  # run HRM core with optional aux
    z = torch.nan_to_num(z)  # clamp latent activations in case upstream recurrence produced non-finite values
    if not need_aux:
      aux = None  # normalize aux when unused
    gate_scale = self.gate_scale.to(dtype=z.dtype, device=z.device)  # align buffer dtype/device with activations
    gate_raw = torch.nan_to_num(self.hrm_gate(z), nan=0.0, posinf=1.0, neginf=0.0)  # compute gating signal and sanitize
    gate = (gate_scale * gate_raw).view(-1, 1, 1)  # apply warmup scaling and broadcast gate.
    gate = torch.clamp(gate, 0.0, 1.0)  # enforce valid interpolation weights
    gate_strength = gate.mean().detach().to(torch.float32).clamp_(0.0, 1.0)  # capture aggregate gate openness in float32
    if self.bridge_type == 'prefix':
      mem_hrm = torch.nan_to_num(self.hrm2dec(z))  # project HRM latent to prefix tokens and sanitize
      mem_base = torch.zeros_like(mem_hrm)  # baseline memory
      mem = (1 - gate) * mem_base + gate * mem_hrm  # blend memories
    else:
      mem_hrm = torch.nan_to_num(self.hrm2dec(z))  # project HRM latent to cross-attention memory and sanitize
      mem_base = torch.zeros_like(mem_hrm)  # baseline memory
      mem = (1 - gate) * mem_base + gate * mem_hrm  # blend memories
    mem = torch.nan_to_num(mem)  # ensure decoder memory stays finite
    logits = self.decoder(decoder_input_ids, mem, attention_mask=dec_attn_mask, memory_mask=None)
    loss = None
    if labels is not None:
      loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)  # compute CE loss
      if self.halting_weight > 0 and aux is not None and 'halt' in aux and aux['halt']:
        halt_stack = torch.stack(aux['halt'], dim=0).squeeze(-1).to(torch.float32)  # stack halting probabilities in float32
        halt_stack = torch.nan_to_num(halt_stack)  # guard against numerical overflow in halting traces
        summed = halt_stack.sum(dim=0)  # sum cycle probabilities
        target = torch.full_like(summed, float(self.halting_target))  # build target tensor
        halt_reg = ((summed - target) ** 2).mean()  # compute regularizer
        if torch.isfinite(halt_reg):  # only apply finite penalties
          effective_weight = torch.tensor(self.halting_weight, dtype=torch.float32, device=halt_reg.device) * gate_strength  # scale penalty using float32 math
          loss = loss + effective_weight * halt_reg  # add weighted halting loss
      if self.deep_supervision and aux is not None and 'z_per_cycle' in aux and len(aux['z_per_cycle']) > 1:
        ds_losses = []  # collect deep supervision losses
        for z_cycle in aux['z_per_cycle'][:-1]:
          mem_cycle = self.hrm2dec(z_cycle)  # build memory for this cycle
          logits_cycle = self.decoder(decoder_input_ids, mem_cycle, attention_mask=dec_attn_mask, memory_mask=None)  # run decoder
          ce_cycle = torch.nn.functional.cross_entropy(logits_cycle.view(-1, logits_cycle.size(-1)), labels.view(-1), ignore_index=-100)  # compute CE
          ds_losses.append(ce_cycle)  # store loss
        if ds_losses and self.ds_weight > 0:
          loss = loss + self.ds_weight * torch.stack(ds_losses).mean()  # add averaged DS loss
    return {'logits': logits, 'loss': loss}
