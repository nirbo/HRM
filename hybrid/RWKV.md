# RWKV Ecosystem Reference

This document summarizes upstream tooling around BlinkDL's RWKV architecture to guide our upcoming RWKV-7 + HRM integration effort. Each entry lists the primary URL and the niche it covers so we can decide where to plug in for training, inference, deployment, and tokenizer support.

## Official Training & Kernels

- **BlinkDL/RWKV-LM** — https://github.com/BlinkDL/RWKV-LM  
  Canonical training repository for RWKV (currently v7 "Goose"). Provides PyTorch training scripts, data recipes, and evaluation utilities. Includes config examples for different parameter scales and integrates Triton kernels for speed.
- **BlinkDL/RWKV-CUDA** — https://github.com/BlinkDL/RWKV-CUDA  
  Standalone CUDA/Triton kernels (depthwise conv, time-mix) that accelerate RWKV forward/backward passes. Useful when extending or rebuilding CUDA ops for custom training loops.
- **RWKV/RWKV-infctx-trainer** — https://github.com/RWKV/RWKV-infctx-trainer  
  Community trainer built on DeepSpeed Stage 3 with backprop-through-time to unlock virtually unbounded context length during training. Includes HF dataset integration and recipes for >1M token windows.
- **JL-er/RWKV-PEFT** — https://github.com/JL-er/RWKV-PEFT  
  Official PEFT toolkit for RWKV (supports RWKV-7). Implements LoRA, DiSHA, PiSSA, and state tuning with detailed VRAM charts. Scripts cover SFT pipelines and operator selection (CUDA/Fused-FLA/Triton).

## Inference & Serving

- **BlinkDL/ChatRWKV** — https://github.com/BlinkDL/ChatRWKV  
  Reference chat stack for RWKV models with web UI, prompt templates, and inference scripts. Demonstrates best practices for running RWKV checkpoints interactively.
- **josStorer/RWKV-Runner** — https://github.com/josStorer/RWKV-Runner  
  Cross-platform desktop + server bundle that manages RWKV downloads, launches accelerated inference backends (custom CUDA kernel), and exposes an OpenAI-compatible API endpoint.
- **RWKV/rwkv.cpp** — https://github.com/RWKV/rwkv.cpp  
  GGML-based C/C++ implementation supporting FP32/FP16 and INT4/5/8 quantized inference on CPU or CUBlas. Includes a Python wrapper and is ideal for lightweight deployment.
- **saharNooby/rwkv.cpp** — https://github.com/saharNooby/rwkv.cpp  
  Original GGML port with similar goals; useful reference when comparing quantization support or building custom bindings.
- **RWKV/rwkv-onnx** — https://github.com/RWKV/rwkv-onnx  
  Converter plus test harness for exporting RWKV (v5+) to ONNX with fp16/fp32 and TensorRT/MPS acceleration support.
- **cryscan/web-rwkv** — https://github.com/cryscan/web-rwkv  
  Pure WebGPU/Rust implementation targeting browser execution; showcases lightweight client-side inference options.

## Tokenizers & Language Bindings

- **RWKV/RWKV-tokenizer-node** — https://github.com/RWKV/RWKV-tokenizer-node  
  Zero-dependency Node.js tokenizer binding compatible with RWKV, GPT-NeoX, and Pythia vocabularies. Useful for JS-based tooling or dataset preprocessing.
- **RWKV/World-Tokenizer-Typescript** — https://github.com/RWKV/World-Tokenizer-Typescript  
  TypeScript tokenizer implementation plus packaging for web projects.
- **RWKV/RWKV-cpp-node** — https://github.com/RWKV/RWKV-cpp-node  
  Node.js native bindings that wrap the C++ runtime for easy integration with serverless or Electron deployments.

## Ecosystem & Community Resources

- **RWKV.com** — https://rwkv.com  
  Central hub with architecture papers, model checkpoints, and ecosystem news.
- **RWKV Wiki** — https://wiki.rwkv.com  
  Community-curated documentation covering concepts, tuning guides, and hardware tips.
- **RWKV Discord** — https://discord.gg/bDSBUMeFpc  
  Active developer community (~9k members) with channels for training support, kernel optimization, and model releases.
- **BlinkDL Twitter** — https://twitter.com/BlinkDL_AI  
  Real-time announcements on RWKV releases, research drops, and kernel updates.

## Complementary Projects to Monitor

- **OpenGVLab/Vision-RWKV** — https://github.com/OpenGVLab/Vision-RWKV  
  Vision adaptation of RWKV (ICLR 2025 Spotlight). Offers insights into multimodal extensions and efficient convolutional front-ends.
- **JL-er/RWKV-PEFT DiSHA Paper** — https://arxiv.org/pdf/2409.15371  
  Research behind the DiSHA adapter (dimension sharding) implemented in RWKV-PEFT, highlighting faster convergence vs standard LoRA.
- **BlinkDL/RWKV-LM/tree/main/RWKV-v7** — https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7  
  Dedicated folder with RWKV-7 Goose scripts, configs, and migration notes.

## Next Steps for Our Integration

1. Clone and study BlinkDL/RWKV-LM and RWKV-PEFT to map the training loop expectations (optimizer, scheduler, state management).  
2. Identify which CUDA operator path (CUDA vs FLA vs Triton) best matches our existing infrastructure.  
3. Evaluate RWKV-infctx-trainer for long-context experiments and determine how to merge HRM outer-loop logging.  
4. Leverage RWKV-Runner or rwkv.cpp for quick inference sanity checks once we attach HRM components.

## Loader Prototype Plan

- Target interface: `RWKV7Encoder` exposing `forward(input_ids, attention_mask)` returning full sequence hidden states plus CLS pooling.
- Implementation outline: reuse `rwkvt` modules from RWKV-PEFT; instantiate `TrainingArgs` with vocab/model dims; copy RWKV forward loop up to `ln_out` to avoid logits head.
- Environment defaults: disable fused kernels (`FUSED_KERNEL=0`) and streaming state modes unless explicitly requested; set `RWKV_TRAIN_TYPE` based on config for standard training.
- Checkpoint handling: load HuggingFace `.pth` via `torch.load`, remap keys to wrapper module; support caching of `ln_out` + block weights.
- Next steps: vendor minimal RWKV-PEFT components into `src/hrm_lm/models/rwkv7_backend.py` for reproducibility, add unit smoke test to ensure hidden outputs match reference logits before head.

## Trainer Gap Analysis

- **Gradient accumulation**: native RWKV-Lightning loop performs one optimizer step per Lightning batch; no accumulate; we must keep our `grad_accum_steps` pipeline when we host RWKV weights.
- **Checkpoint cadence**: RWKV trainer writes epoch-based `.pth` snapshots without artifact copies; we need to retain our step-indexed checkpoints, tokenizer/config snapshots, and retention limit.
- **Validation/Early stop**: RWKV scripts skip evaluation altogether; our loop must schedule validation sweeps, compute metrics, and honor `eval_loss_patience` & grace windows.
- **Non-finite guardrails**: existing RWKV callback lacks NaN detection or batch dumps; we should keep our runtime guard + diagnostic payloads.
- **Dataset support**: RWKV expects `.binidx` streams; we rely on JSONL + streaming/backfilling. Reusing our loader requires bridging `RWKV-LM` binidx support or converting datasets upfront.
- **Per-parameter LR scaling**: RWKV optimizers set custom groups (1x/2x + decay). If we stick with our `pytorch_optimizer` integration we need to replicate these groups or allow config to import them.
- **Grad checkpoint / CUDA graph**: RWKV uses manual gradient checkpoint toggles harming CUDA-graphs; our trainer should mark `supports_cuda_graphs=False` for RWKV backend and optionally surface grad-CP toggles.
- **Environment flags**: RWKV modules rely on env vars (`RWKV_MY_TESTING`, `FUSED_KERNEL`, `RWKV_TRAIN_TYPE`); we must manage them explicitly during model init to avoid global side-effects.

## Integration Notes

- Set `model.encoder.backend: rwkv7` in the training config to activate the new wrapper.
- Provide `model.encoder.encoder_cfg.checkpoint_path` pointing to a RWKV-7 `.pth` file; if omitted, the loader searches for `models/blinkdl-rwkv7-g1a-1.5b/rwkv-final.pth` by default.
- Optional overrides include `head_size_a`, `head_size_divisor`, `dim_att`, `dim_ffn`, and precision settings mirroring RWKV-PEFT `TrainingArgs`.
- Ensure the CUDA toolchain is available so RWKV-PEFT can JIT compile `cuda/wkv7_cuda.cu` kernels on first import.
