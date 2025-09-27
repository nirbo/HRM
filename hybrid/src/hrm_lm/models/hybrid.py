# Unified HRM+LM
import torch
import torch.nn as nn
from .encoder import LMEncoder
from .hrm_core import HRMCore
from .bridges import (
  PromptToHRMBridge,
  HRMToPrefixBridge,
  HRMToCrossAttnBridge,
  HRMGate,
)
from .decoder import LMDecoder


class HRMLanguageModel(nn.Module):
  def __init__(
    self,
    vocab_size,
    d_model,
    enc_layers,
    dec_layers,
    max_enc_len,
    max_dec_len,
    hrm_cfg,
    bridge_cfg,
    enc_backend="transformer",
    encoder_cfg=None,
  ):
    super().__init__()
    self.encoder = LMEncoder(
      vocab_size,
      d_model,
      enc_layers,
      max_enc_len,
      backend=enc_backend,
      encoder_cfg=encoder_cfg,
    )
    self.prompt2hrm = PromptToHRMBridge(d_model)
    use_halting = hrm_cfg.get("use_halting", False)
    self.hrm = HRMCore(
      d_model=d_model,
      h_len=hrm_cfg["h_len"],
      l_len=hrm_cfg["l_len"],
      h_layers=hrm_cfg["h_layers"],
      l_layers=hrm_cfg["l_layers"],
      h_cycles=hrm_cfg["h_cycles"],
      l_steps=hrm_cfg["l_steps"],
      approx_grad=hrm_cfg["approx_grad"],
      out_dim=hrm_cfg["out_dim"],
      use_halting=use_halting,
    )
    self.use_halting = use_halting
    self.halting_weight = hrm_cfg.get("halting_weight", 0.0)
    default_target = 1.0 if use_halting else float(hrm_cfg.get("h_cycles", 0))
    self.halting_target = float(hrm_cfg.get("halting_target", default_target))
    self.deep_supervision = hrm_cfg.get("deep_supervision", False)
    self.ds_weight = hrm_cfg.get("ds_weight", 0.0)
    self.bridge_type = bridge_cfg["type"]
    if self.bridge_type == "prefix":
      self.hrm2dec = HRMToPrefixBridge(d_model, bridge_cfg["prefix_len"])
    else:
      self.hrm2dec = HRMToCrossAttnBridge(d_model)
    self.decoder = LMDecoder(vocab_size, d_model, dec_layers, max_dec_len)
    self.hrm_gate = HRMGate(d_model)
    gate_scale_cfg = float(hrm_cfg.get("gate_scale", 1.0))
    gate_bias_cfg = float(hrm_cfg.get("gate_bias", 0.0))
    self.register_buffer(
      "gate_scale_base", torch.tensor(gate_scale_cfg, dtype=torch.float32)
    )
    self.register_buffer(
      "gate_scale", torch.tensor(gate_scale_cfg, dtype=torch.float32)
    )
    self.register_buffer(
      "gate_bias", torch.tensor(gate_bias_cfg, dtype=torch.float32)
    )
    self.supports_cuda_graphs = bool(getattr(self.encoder, 'supports_cuda_graphs', True))  # expose encoder capture capability to the trainer

  def forward(
    self,
    input_ids,
    decoder_input_ids,
    enc_attn_mask=None,
    dec_attn_mask=None,
    labels=None,
  ):
    enc_h, cls, enc_aux = self.encoder(input_ids, enc_attn_mask)
    x = self.prompt2hrm(cls)
    need_aux = self.halting_weight > 0 or self.deep_supervision
    z, aux = self.hrm(x, return_all=need_aux)
    z = torch.nan_to_num(z)
    if not need_aux:
      aux = None

    gate_scale = self.gate_scale.to(dtype=z.dtype, device=z.device)
    gate_bias = self.gate_bias.to(dtype=z.dtype, device=z.device)
    gate_raw = torch.nan_to_num(
      self.hrm_gate(z), nan=0.0, posinf=1.0, neginf=0.0
    )
    gate = (gate_scale * gate_raw + gate_bias).view(-1, 1, 1)
    gate = torch.clamp(gate, 0.0, 1.0)
    gate_strength = (
      gate.mean().detach().to(torch.float32).clamp_(0.0, 1.0)
    )
    metrics = {
      "gate_mean": gate_strength.item(),
      "gate_min": gate.min().detach().item(),
      "gate_max": gate.max().detach().item(),
    }

    mem_hrm = torch.nan_to_num(self.hrm2dec(z))
    mem_base = torch.zeros_like(mem_hrm)
    mem = (1 - gate) * mem_base + gate * mem_hrm
    mem = torch.nan_to_num(mem)
    logits = self.decoder(
      decoder_input_ids,
      mem,
      attention_mask=dec_attn_mask,
      memory_mask=None,
    )

    loss = None
    if labels is not None:
      loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
      )
      if (
        self.halting_weight > 0
        and aux is not None
        and "halt" in aux
        and aux["halt"]
      ):
        halt_stack = torch.stack(aux["halt"], dim=0).squeeze(-1).to(
          torch.float32
        )
        halt_stack = torch.nan_to_num(halt_stack)
        summed = halt_stack.sum(dim=0)
        metrics.update(
          {
            "halt_sum_mean": summed.mean().detach().item(),
            "halt_sum_min": summed.min().detach().item(),
            "halt_sum_max": summed.max().detach().item(),
            "halt_cycle_mean": halt_stack.mean().detach().item(),
            "halt_target": float(self.halting_target),
          }
        )
        target = torch.full_like(summed, float(self.halting_target))
        halt_reg = ((summed - target) ** 2).mean()
        if torch.isfinite(halt_reg):
          effective_weight = (
            torch.tensor(
              self.halting_weight,
              dtype=torch.float32,
              device=halt_reg.device,
            )
            * gate_strength
          )
          loss = loss + effective_weight * halt_reg

      if (
        self.deep_supervision
        and aux is not None
        and "z_per_cycle" in aux
        and len(aux["z_per_cycle"]) > 1
      ):
        ds_losses = []
        for z_cycle in aux["z_per_cycle"][:-1]:
          mem_cycle = self.hrm2dec(z_cycle)
          logits_cycle = self.decoder(
            decoder_input_ids,
            mem_cycle,
            attention_mask=dec_attn_mask,
            memory_mask=None,
          )
          ce_cycle = torch.nn.functional.cross_entropy(
            logits_cycle.view(-1, logits_cycle.size(-1)),
            labels.view(-1),
            ignore_index=-100,
          )
          ds_losses.append(ce_cycle)
        if ds_losses and self.ds_weight > 0:
          loss = loss + self.ds_weight * torch.stack(ds_losses).mean()

    if enc_aux is not None and loss is not None and getattr(self.encoder, 'moe_aux_weight', 0.0) > 0:
      aux_weight = self.encoder.moe_aux_weight
      loss = loss + aux_weight * enc_aux
      metrics['moe_aux_loss'] = (aux_weight * enc_aux).detach().item()

    return {"logits": logits, "loss": loss, "metrics": metrics}
