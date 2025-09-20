# FP8 Training Landscape (September 20, 2025)

## 1. Summary of Current Support
- **PyTorch Core (2.8.0+)**
  - Ships `torch.float8_e4m3fn` and `torch.float8_e5m2` dtypes (initially in 2.2) and integrates float8-aware ops through `torch._scaled_mm`, `torch.compile`, and FSDP2/DTensor pipelines. 2.8 documentation and release collateral reference float8 end-to-end flows via TorchAO and TorchTitan.
  - FSDP2 + DTensor provide float8 communication primitives (e.g., all-gathers) highlighted in the Nov 2024 PyTorch blog *“Supercharging Training using float8 and FSDP2”*, demonstrating up to **1.43–1.51×** pre-training speedups on Llama 3.1 70B/405B compared with BF16 (tensorwise scaling) while keeping accuracy parity.
  - The Apr 2025 PyTorch/Crusoe blog reports **34–43 % throughput gains** at 2 K H200 scale using the new **rowwise** float8 recipe, confirming stable convergence over 15 k iterations and multi-day runs.
- **TorchAO 0.13 (PyTorch AO)**
  - TorchAO README (May 2025) positions float8 training/inference as first-class features; tutorials *(Part 1) Pre-training with float8* show Torchtitan integration, recipes (tensorwise vs rowwise), and CLI flags (`--model.converters="float8"`). 8-H100 experiments deliver **1.11–1.21× TPS** improvements at unchanged memory footprints.
  - Float8 recipes require `torch.compile`, support FSDP2 across distributed topologies, and expose APIs for user code (`torchao.float8`).
- **Ecosystem Integrations**
  - **TorchTitan**: official PyTorch pre-training stack with float8 recipes, deterministic benchmarking, and distributed checkpointing.
  - **Axolotl (Sept 2025 docs)**: FP8 mixed-precision option (`fp8: true`) marked experimental; requires Hopper+/Blackwell GPUs, PyTorch ≥2.7, TorchAO, CUDA ≥12.4. Optional flag enables FSDP float8 all-gathers.
  - **Transformers / vLLM / SGLang / Torchtune**: TorchAO backend offers float8 fine-tuning recipes (rowwise/tensorwise) for LLaMA family.
  - **Transformer Engine & MXFP8**: NVIDIA’s June 2025 blog introduces MXFP8 (micro-scaling) on Blackwell GPUs to reduce quantization error; TE orchestrates mixed MXFP8/FP8/BF16 compute automatically.
- **Academic Results**
  - *FP8-LM* (Dec 2023) proved 75 % faster GPT-175B training with an FP8 framework that pushes gradients, optimizer states, and communication to 8-bit, reducing real memory by 39 % without hyperparameter changes.

## 2. Hardware Considerations
- **Hopper/H100 & Blackwell/H200**: Full FP8 Tensor Core support with FP16 or FP32 accumulation paths; rowwise scaling (power-of-two factor per row) improves precision for large-scale training.
- **RTX 40/50 Series (5090)**: Reddit (Jan 2025) notes NVIDIA halved FP8 throughput *with FP32 accumulate* in Blackwell consumer GPUs (e.g., 5090 advertises 419 TFLOPs FP8 FP32-acc vs 838 TFLOPs FP16-acc). FP8 training requiring FP32 accum may therefore run at half speed compared to datacenter SKUs. FP16-accumulate path still advertises full rate but jeopardizes training fidelity.
- **Software Stack**: For float8 training today you need:
  - PyTorch ≥2.7 (preferably 2.8 nightly/stable) with `torch.compile`, FSDP2, DTensor.
  - TorchAO ≥0.12/0.13 for float8 recipes.
  - CUDA 12.3+ (rowwise uses 12.4+) and appropriate driver.
  - Transformer Engine optional for MXFP8 or per-channel scaling.

## 3. FP8 Workflows Available Now
1. **TorchTitan + TorchAO** (recommended path)
   - Launch with `--model.converters="float8"` and choose `--float8.recipe_name=rowwise|tensorwise`.
   - Gains: 1.1–1.5× throughput vs BF16 on H100/H200; deterministic mode validated convergence on up to 2K GPUs.
   - Limitations: Multi-GPU only (≥2 GPUs), FP8 attention still BF16 (work-in-progress).
2. **Native Integration in Custom Trainer**
   - Use `torchao.float8` APIs to wrap linear layers, provide scaling strategy, and ensure `torch.compile` is enabled.
   - Float8 all-gathers require FSDP2 nightly/stable (PyTorch 2.8) and may need environment variable `TORCHAO_FLOAT8=1` depending on build.
3. **Axolotl FP8 configs**
   - YAML snippet enables float8 end-to-end for instruction tuning or SFT; optional FSDP float8 all-gather flag ensures communication stays in FP8.
4. **Transformer Engine / MXFP8**
   - Suitable if you rely on TE (Megatron-LM, NeMo). TE handles dynamic scaling, fallback, and mixed accumulate paths.

## 4. Stability & Edge Cases
- **Precision Strategy**: Rowwise scaling gives better numerical stability (preferred for long intra-run stability); tensorwise maximizes throughput but introduces higher quantization noise.
- **Attention & Non-linear Ops**: Most pipelines still compute attention in BF16 (mentioned in Nov 2024 blog). Work is ongoing to push attention to FP8; expect incremental rollouts.
- **FP32 Accumulate Requirement**: For reliable training of deep transformers, prefer FP8 with FP32 accumulate. Consumer GPUs (5090) throttling this path means datacenter GPUs (H100/H200/B100) remain the pragmatic choice for full-speed FP8 training runs.
- **API Maturity**: Medium article (May 2024) stressed PyTorch native FP8 as experimental. By 2025, TorchAO/TorchTitan flows are marked “preview/experimental but production-evaluated” on Meta & IBM deployments. Expect API churn; pin to tested commits or TorchAO release tags.

## 5. Feasibility for Our Project (RTX 5090 single-node training)
- **Pros**
  - PyTorch 2.8 + TorchAO 0.13 give us ready-made float8 recipes integrated with FSDP2.
  - Our dataset (≈1.66 M sequences @512) is within memory bounds; FP8 could reduce memory pressure vs BF16, enabling larger context/batch.
- **Cons / Caveats**
  - RTX 5090’s FP8 FP32-accumulate throughput is capped at ~419 TFLOPs, ~50 % of FP16-acc path; expect smaller speedups vs BF16 (~20–30 %?) compared with H100 (30–50 %).
  - Training stability may suffer if forced onto FP16 accumulate; would necessitate extensive validation.
  - FP8 tooling is still officially “experimental”; debugging infra (profilers, hooks) less mature than BF16.

**Recommendation**
1. Keep BF16 as baseline for the production run; gather speed/quality metrics.
2. Prototype FP8 using TorchAO tensorwise + rowwise recipes on 5090 to benchmark throughput and loss behaviour. Monitor for divergence; fall back to BF16 if rowwise still unstable.
3. For large-scale runs (multi-GPU or cloud), prefer renting H100/H200/B100 nodes to exploit full FP8 advantage (FSDP2 + float8 all-gather).
4. Track upstream: TorchAO releases, PyTorch release notes, NVIDIA driver updates (potentially unlocking FP32-acc FP8 on GeForce) and TE’s MXFP8 support.

## 6. Key References
- PyTorch Blog (Nov 25 2024): *Supercharging Training using float8 and FSDP2* — IBM/Meta report 1.43–1.51× speedups with FSDP2 + TorchAO tensorwise float8.
- PyTorch Blog (Apr 28 2025): *Accelerating Large Scale Training and Convergence with PyTorch Float8 Rowwise on Crusoe 2K H200s* — Rowwise scaling for float8, 34–43% throughput gain.
- TorchAO README (May 2025) + Tutorials: Float8 training/inference flows, torchtitan integration, APIs.
- Medium (May 21 2024): *PyTorch Native FP8 Data Types* — Introduces torch.float8 dtypes, warns experimental status and limited ops in 2.3.
- Axolotl Docs (2025): FP8 configuration requirements (PyTorch ≥2.7, TorchAO, H100/H200, CUDA 12.4+).
- NVIDIA Technical Blog (Jun 4 2025): FP8 & MXFP8 fundamentals on Blackwell GPUs.
- FP8-LM Paper (arXiv 2310.18313v2): Framework reducing GPT-175B training time by 75% vs BF16 with full FP8 gradients and optimizer states.
- Reddit /r/LocalLLaMA (Jan 2025): Reports FP8 FP32-acc throughput halved on RTX 40/50 consumer GPUs.
