# CUDA Optimization Design Notes

This document tracks GPU-focused optimisation work for the HRM/MoE trainer. It summarises the highest-impact initiatives, their status, expected gains, and concrete action plans so we can execute methodically.

## At-a-Glance Roadmap

| Initiative | Status | Priority | Expected Benefit | Key Requirements |
| --- | --- | --- | --- | --- |
| Nsight Systems + Nsight Compute profiling baseline | Implemented (testing pending) | **Critical** | Establish kernel-level timeline and hotspot ranking to guide all downstream work | Install Nsight tooling, integrate trace capture hooks in trainer, capture representative training/eval runs |
| FP8 training via TransformerEngine (Blackwell-ready) | Implemented (testing pending) | **Critical** | 1.4×–2.0× throughput gains; modest memory savings once activations dominate | Hopper/Blackwell GPU, CUDA ≥12.2, TransformerEngine install, FP8 scaling calibration, checkpoint scale metadata |
| CUDA graph capture of forward/backward/optimizer | Implemented (testing pending) | High | Removes Python launch overhead; NVIDIA reports 1.1×–1.7× overall speedups and up to 5× within graphed regions | Static shapes, no host syncs during capture, persistent buffers, post-capture regression test |
| CUTLASS expert feed-forward kernels (`nvidia-cutlass`) | Planned | High | Tensor-core optimised GEMMs for MoE/FFN blocks; potential 20–40% speedup on expert matmuls | CUTLASS python bindings, Nsight hotspot data, custom PyTorch extension glue |
| cuSPARSELt structured sparsity (2:4) | Planned | Medium-High | Exploit structured sparse weights for additional 1.3×–1.6× GEMM speedups if experts adopt 2:4 sparsity | Prune/finetune to 2:4 pattern, integrate cuSPARSELt API, sparsity-aware optimizer updates |
| CuPy-based fused kernels (`cupy-cuda12x`) | Planned | Medium | Rapid prototyping of fused HRM gate / halting ops with zero-copy PyTorch↔CuPy interop | Install matching CuPy wheel, autograd Function wrappers, correctness tests vs. ATen ops |
| Allocator & tensor-core tuning (TF32, cudaMallocAsync) | In progress | Medium | Improve stability + memory utilisation; prerequisite hygiene for graphs/FP8 | Maintain TF32 config, monitor memory, evaluate async allocator / pluggable backends |
| Data pipeline acceleration (NVIDIA DALI / RAPIDS) | Backlog | Medium-Low | Offload heavy preprocessing or future multimodal augmentation to GPU to keep trainers saturated | Identify CPU bottlenecks, integrate DALI or RAPIDS loaders for relevant datasets |

## 1. Nsight Profiling Baseline (Critical)

**Current status**  
- Trainer emits NVTX ranges around data loading, forward/backward, optim step, and eval loops (gated by config).  
- `scripts/capture_nsight.py` wraps `nsys`/`ncu` to launch traces with a single command.  
- Ready for trace collection once Nsight CLI tools are installed.

**Why it matters**  
We still need quantitative evidence before deeper kernel work. Nsight Systems surfaces timeline-level behaviour (kernel order, launch gaps, memcpy overlap) while Nsight Compute provides per-kernel metrics such as occupancy, tensor-core utilisation, and dram throughput.[1][2]

**Next actions**
1. Install Nsight Systems (`nsys`) and Nsight Compute (`ncu`) on the training host; verify CLI operation without GUI.
2. Enable NVTX ranges via config (`profiling.nvtx.enabled=true`) and capture representative 200–500 step windows to keep trace sizes manageable.
3. Capture baseline traces for the current BF16 + MoE configuration; archive `.nsys-rep` / `.ncu-rep` artefacts under `profiling/` with config + commit metadata.
4. Summarise hotspots (top kernels, SM utilisation, dram bandwidth) in a short markdown report to guide optimisation priorities.
5. Re-run profiling after each major optimisation (FP8, CUTLASS, sparsity) to track gains and catch regressions early.

**Deliverables**
- `profiling/README.md` documenting capture commands.
- Baseline Nsight Systems and Nsight Compute trace artefacts.
- Hotspot digest feeding directly into downstream work.

## 2. FP8 Training with TransformerEngine (Critical)

**Current status**  
- Transformer encoder FFNs switch to `TransformerEngine.Linear` when `train.mixed_precision=fp8`; MoE experts automatically fall back to BF16 to avoid TE’s FP8 shape constraints.  
- Training loop pads batches to the configured `seq_len` so FP8 kernels see token counts divisible by 8/16, manages warmup/`DelayedScaling`, and records checkpoint metadata automatically.  
- CUDA graphs are auto-disabled for FP8 runs (TE is not capture-safe yet). Logging announces the FP8 activation point and now surfaces both clipped and raw gradient norms.  
- VRAM usage is broadly comparable to BF16 (FP8 activations + BF16 master weights/optimizer states); throughput improves thanks to higher tensor-core utilisation.

**Why now**  
RTX 5090 (Blackwell) exposes FP8 tensor cores. TransformerEngine (TE) demonstrates 1.4×–2.0× throughput gains and roughly 40–60% activation memory reductions when transformers shift from BF16 to FP8 on Hopper/Blackwell GPUs.[3][4]

**Prerequisites**
- CUDA ≥12.2, PyTorch ≥2.1.
- `pip install transformer-engine` matching the CUDA toolkit.
- Runtime guard: `torch.cuda.get_device_capability() ≥ (12, 0)`.

**Operational checklist**
1. Install `transformer-engine` matching the CUDA toolkit if it is not already present.
2. Set `train.mixed_precision: fp8` (and adjust `train.fp8.*` as needed); keep MoE disabled or accept BF16 experts until TE supports routed FP8 kernels.
3. Watch for the “FP8 autocast enabled…” banner and monitor raw gradient norms to confirm healthy scaling.
4. Profile throughput/memory (Nsight Systems/Compute + `torch.cuda.memory_summary`) against the BF16 baseline; expect speedups but only modest VRAM savings.

**Deliverables**
- Updated trainer/config enabling FP8 safely on Blackwell.
- Checkpoint schema storing TE scaling metadata.
- Regression tests and documentation for FP8 workflows.

## 3. CUDA Graph Capture (Implemented – testing pending)

**Status**  
`train.use_cuda_graphs` now attempts capture but falls back to eager execution when the encoder flags incompatibility (MoE, Mamba2) or capture throws. This prevents the crash observed previously while keeping graphs available for transformer-only runs.[5][6]

**Follow-up**
- During the next training session confirm the fallback warning appears and throughput matches pre-change numbers.
- Schedule a transformer-only experiment (graph-compatible) once higher-priority items land to measure real speedups.

## 4. CUTLASS Expert Kernels (High)

**Why**  
MoE experts and dense FFNs dominate FLOPs. CUTLASS’s Python bindings expose tensor-core optimised GEMMs/grouped GEMMs so we can specialise high-payoff shapes without hand-written C++.[7]

**Plan**
1. Use Nsight hotspot data to catalogue expert matmul shapes (batch size, hidden size, expert count).
2. Prototype CUTLASS kernels, wrap them as PyTorch custom ops, and benchmark vs. cuBLAS/ATen.
3. Integrate the fastest kernels with safe fallbacks for unsupported shapes; document accuracy/throughput deltas.

## 5. cuSPARSELt Structured Sparsity (Medium-High)

**Why**  
Blackwell tensor cores accelerate 2:4 structured sparse GEMMs. cuSPARSELt offers weight compression, validation, and sparse matmul APIs with reported 1.3×–1.6× gains once the pattern is enforced.[8]

**Plan**
1. Investigate magnitude pruning or sparse-aware fine-tuning to push MoE experts toward the 2:4 pattern.
2. Integrate cuSPARSELt via a custom extension; ensure compression/decompression overhead is amortised across batches.
3. Compare accuracy vs. dense baselines before promoting to default configs.

## 6. CuPy Fused Kernels (Medium)

**Why**  
CuPy enables rapid CUDA kernel prototyping in Python with zero-copy tensor sharing, ideal for HRM gate/halting fusion or routing-statistic kernels before porting to CUTLASS/C++.[9]

**Plan**
1. Install `cupy-cuda12x` matching the system toolkit.
2. Implement candidate fused kernels via `cupy.RawKernel` inside autograd `Function`s.
3. Validate gradients numerically; upstream wins into more permanent implementations when justified.

## 7. Allocator & Tensor-Core Hygiene (Supporting)

**Why**  
Stable memory behaviour and tensor-core settings underpin CUDA graphs and FP8. PyTorch exposes TF32 toggles and optional allocators (`cudaMallocAsync`, pluggable NCCL allocators).[5]

**Actions**
- Keep TF32 enabled (already defaulted via `train.enable_tf32`).
- Monitor `torch.cuda.memory_summary()` on long runs; experiment with `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` if fragmentation appears.
- Evaluate pluggable allocators once multi-GPU/NCCL workloads arrive.

## 8. Data Pipeline Acceleration (Backlog)

**Why**  
If future datasets add heavy CPU preprocessing or multimodal augmentation, GPU-side loaders (NVIDIA DALI, RAPIDS cuDF/cuML) can keep the trainer saturated.[10][11]

**Plan**
1. Benchmark dataloader CPU utilisation after FP8 rollout. If it exceeds ~80%, prototype DALI pipelines for the busiest modalities.
2. Adopt RAPIDS cuDF/cuML for large-scale preprocessing where appropriate.
3. Gate additions behind config flags so lightweight text-only runs remain unaffected.

## 7. Allocator & Tensor-Core Hygiene (Supporting)

**Current status**  
- Trainer and generator now set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` by default, reducing fragmentation during long FP8 runs.  
- TF32 remains enabled by default for BF16/FP8 paths; MoE continues in BF16 where needed.  
- No asynchronous allocator experiments yet (cudaMallocAsync / pluggable NCCL allocators still backlog items).

**Next actions**
1. Monitor long FP8 runs for residual fragmentation; if issues remain, prototype cudaMallocAsync or the PyTorch pluggable allocator API.  
2. When multi-GPU training arrives, profile NCCL behaviour and decide whether a custom allocator is warranted.

## Phased Execution Plan (Updated)

1. **Phase 0 – Profiling Foundation**
   - Instrument trainer with Nsight Systems/Compute hooks; archive baseline traces and hotspot report.
2. **Phase 1 – FP8 Rollout**
   - Implement TransformerEngine integration, warmup, checkpoint metadata, and regression tests.
3. **Phase 2 – CUTLASS + Graph Validation**
   - Optimise expert matmuls with CUTLASS and re-evaluate CUDA graph benefits on compatible models.
4. **Phase 3 – Sparsity & Kernel Fusion**
   - Explore cuSPARSELt structured sparsity and CuPy fused kernels guided by updated profiling data.
5. **Phase 4 – Memory & Pipeline Enhancements**
   - Address allocator tuning, consider DALI/RAPIDS integration, and maintain telemetry hygiene.

## References

1. *Nsight Systems User Guide.* https://docs.nvidia.com/nsight-systems/
2. *Nsight Compute User Guide.* https://docs.nvidia.com/nsight-compute/
3. *TransformerEngine GitHub Repository.* https://github.com/NVIDIA/TransformerEngine
4. *“Transformer Engine Accelerates FP8 Training on Hopper GPUs,”* NVIDIA Technical Blog (2023). https://developer.nvidia.com/blog/transformer-engine-accelerates-fp8-training-on-hopper-gpus/
5. *PyTorch 2.8 CUDA semantics – CUDA Graphs & allocator notes.* https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs
6. *“Accelerating PyTorch with CUDA Graphs,”* PyTorch/NVIDIA Blog (Nov 2024). https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
7. *CUTLASS Python README.* https://github.com/NVIDIA/cutlass/blob/main/python/README.md
8. *cuSPARSELt Documentation.* https://docs.nvidia.com/cuda/cusparselt/index.html
9. *CuPy User Guide – PyTorch Interoperability.* https://docs.cupy.dev/en/stable/user_guide/interoperability.html
10. *NVIDIA DALI Documentation.* https://docs.nvidia.com/deeplearning/dali/
11. *RAPIDS cuDF User Guide.* https://docs.rapids.ai/api/cudf/stable/
12. *PyTorch 2.8.0 Release Notes.* https://github.com/pytorch/pytorch/releases/tag/v2.8.0

## CUDA 13.x Support Status (Research Summary)

As of PyTorch **2.8.0** (latest stable release, August 2025), official binaries are published only for CUDA 11.x and CUDA 12.x toolkits. The 2.8 release notes discuss CUDA 12.8/12.9 builds and even call out a Windows-specific workaround for CUDA 12.9.1, but **no CUDA 13 wheels are offered** in the primary “Get Started” selector or release artefacts.[12]

NVIDIA/PyTorch maintainers are actively preparing CUDA 13 coverage, but the work is still in-flight:

- Pull request **[#161663](https://github.com/pytorch/pytorch/pull/161663)** (“Add CUDA 13.0 Windows build”) landed in the main branch to enable Windows CI builds for CUDA 13, but this targets future releases and is not part of 2.8.0 stable artefacts.
- Issue **[#163983](https://github.com/pytorch/pytorch/issues/163983)** (“Add more CI tests for CUDA 13.0”) remains open, noting that CUDA 13 runs only in periodic CI and requesting expanded coverage for PyTorch 2.10.

**Implication for RTX 5090 (Blackwell):** the GPU hardware supports CUDA 13 features, but until PyTorch publishes official CUDA 13 wheels we should stay on the CUDA 12.9 packages (e.g., the `+cu129` builds) or build PyTorch from source against the CUDA 13 toolkit if experimentation is required. Revisit once the upstream CI work ships in an official release.
