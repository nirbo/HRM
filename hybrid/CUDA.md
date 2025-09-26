# CUDA Optimization Design Notes

This document summarizes the CUDA-accelerated execution paths we can layer onto the HRM/MoE trainer, the expected impact, and concrete plans for integration. Each section captures evidence from NVIDIA/PyTorch sources, prerequisites, and implementation tasks so we can prioritize and schedule the work.

## At-a-Glance Roadmap

| Optimization | Priority | Expected Benefit | Key Requirements |
| --- | --- | --- | --- |
| CUDA Graph capture of forward/backward/optimizer | **High** | 1.1×–1.7× overall speedups on NVIDIA benchmarks once CPU launch overhead is removed; up to 5× faster for graphed regions of Mask R-CNN | Static tensor shapes, removal of CPU syncs, persistent buffers reused per replay |
| CuPy-based fused kernels (via `cupy-cuda12x`) | Medium | Zero-copy PyTorch↔CuPy interop for bespoke kernels; rapid prototyping of fused HRM/MoE ops entirely in Python | Install matching CuPy wheel; manage memory via DLPack; wrap ops in autograd Functions |
| CUTLASS Python interface (`nvidia-cutlass`) | Medium | High-performance tensor-core GEMMs, grouped GEMMs, and fused epilogues without writing C++; emit PyTorch extensions when ready | Install CUTLASS Python bindings; profile expert matmuls; export kernels for deployment |
| Tensor-core & allocator tuning (TF32, CUDA allocators) | **High** | Immediate gains via tensor cores (TF32) and improved memory behavior; prerequisites for stable CUDA Graph capture | Enable TF32 in config; consider async allocator / custom pools; monitor with Nsight |

## 1. CUDA Graphs (High Priority)

**What & Why**  
CUDA graphs capture an entire training step and replay it with a single `cudaGraphLaunch`, eliminating Python/C++ launch overhead and keeping kernels tightly packed. NVIDIA reports graphed portions of Mask R-CNN dropping from 31 ms to 6 ms (≈5×) and an overall **1.7× end-to-end speedup** after graphing the backbone. Their max-scale BERT run saw **1.12× speedup** once tensor shapes were forced static and CPU synchronizations removed.[1][2]

**Constraints**
- Static shapes for tensors within the captured region (HRM/MoE already operates on fixed batch/seq lengths).
- No CPU-side synchronizations or allocator calls inside the graphed span.
- RNG state and optimizer buffers must reuse the same storage each replay.

**Implementation Plan**
1. **Refactor the training loop** so forward, loss, backward, and optimizer step occur in a dedicated `run_step()` function with no logging/printing. Verified by unit tests.
2. **Warm-up on a side stream** (2–3 iterations) to materialize parameter gradients and optimizer state. Use the same pattern as PyTorch’s official example (see code below).
3. **Allocate persistent “static” tensors** for inputs/targets/metrics. During training, copy each new batch into these tensors before replaying the graph.
4. **Capture** the step with `torch.cuda.graph(graph)` and store handles to static outputs (loss, metrics, grads) for inspection.
5. **Replay** in the main loop (`graph.replay()`), then run any host-side logging/validation outside the graphed path.
6. **Validation:** compare eager vs. graphed loss for a fixed seed/batch to confirm identical updates.
7. **Profiling:** run Nsight Systems before/after to verify CPU launch gaps disappear.

**Code Skeleton**
```python
# Warm-up on a side stream
warmup_stream = torch.cuda.Stream()
warmup_stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(warmup_stream):
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = run_step(static_input, static_target)
        loss.backward()
        optimizer.step()
torch.cuda.current_stream().wait_stream(warmup_stream)

# Capture
cuda_graph = torch.cuda.CUDAGraph()
optimizer.zero_grad(set_to_none=True)
with torch.cuda.graph(cuda_graph):
    static_loss = run_step(static_input, static_target)
    static_loss.backward()
    optimizer.step()

# Replay each batch
for batch_input, batch_target in loader:
    static_input.copy_(batch_input)
    static_target.copy_(batch_target)
    cuda_graph.replay()
    log_metrics(static_loss.item())
```

## 2. CuPy (`cupy-cuda12x`) for Custom Kernels (Medium Priority)

**What & Why**  
CuPy provides a NumPy-like interface whose arrays interoperate with PyTorch tensors via `__cuda_array_interface__` and DLPack. This allows **zero-copy sharing** of tensor storage and quick prototyping of custom CUDA kernels (`cupy.RawKernel`) inside autograd Functions. The CuPy docs show direct PyTorch↔CuPy conversions and stream/memory-pool sharing so the two libraries coexist without extra copies.[3][4]

**Use Cases in HRM/MoE**
- Fused HRM gate computation or halting regularizer to reduce launch counts.
- Custom MoE routing statistics or sparse reductions.
- Experimental kernels prior to implementing a C++/CUTLASS extension.

**Implementation Plan**
1. Install the wheel matching our CUDA runtime: `pip install cupy-cuda12x` (CUDA 12.x).[4]
2. Wrap target logic in an autograd `Function` that uses CuPy RawKernels for forward/backward (see below). Validate gradients by comparing to PyTorch reference implementations.
3. Share the PyTorch memory allocator/streams using `pytorch-pfn-extras` if necessary to avoid extra allocations.[3]
4. Profile end-to-end to ensure kernel JIT compilation occurs once (e.g., compile at module import).

**Code Skeleton**
```python
import cupy as cp
import torch

forward_kernel = cp.RawKernel(r"""
extern "C" __global__ void gate_kernel(const float* x, float* y, int n) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n) y[tid] = x[tid] > 0.f ? x[tid] : 0.f;
}
""", "gate_kernel")

class CuPyGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        cupy_x = cp.asarray(input.detach())
        cupy_y = cp.empty_like(cupy_x)
        n = cupy_x.size
        threads = 128
        blocks = (n + threads - 1) // threads
        forward_kernel((blocks,), (threads,), (cupy_x, cupy_y, n))
        ctx.save_for_backward(input)
        return torch.from_dlpack(cupy_y)
```

## 3. CUTLASS Integration (Medium Priority)

**What & Why**  
CUTLASS is NVIDIA’s templated CUDA GEMM library with tensor-core optimized kernels. The official **CUTLASS Python interface (`nvidia-cutlass`) lets us compile and run kernels from Python** and “simplifies exporting CUTLASS kernels to framework extensions (e.g., PyTorch CUDA extensions).”[5] This is ideal when MoE experts need specialized matmuls or fused epilogues beyond what cuBLAS/ATen provides.

**Implementation Plan**
1. Install CUTLASS Python bindings inside our development container (`pip install nvidia-cutlass`).
2. Prototype GEMM shapes matching MoE experts (d_model × ff dim) and benchmark vs. PyTorch/FlashAttention kernels.
3. If gains warrant, emit the kernels as a PyTorch CUDA extension using CUTLASS’s emitters (`cutlass_cppgen`) and integrate them behind a feature flag.
4. Profile with Nsight Compute to verify tensor-core utilization and occupancy.

**Example (from CUTLASS README)**
```python
import cutlass
import numpy as np

plan = cutlass.op.Gemm(element=np.float16, layout=cutlass.LayoutType.RowMajor)
A, B, C, D = [np.ones((1024, 1024), dtype=np.float16) for _ in range(4)]
plan.run(A, B, C, D)
```

## 4. Tensor-Core & Allocator Settings (High Priority)

**TF32 Enablement**  
Torch defaults to disabling TF32 matmuls. Enabling it (`torch.backends.cuda.matmul.allow_tf32 = True`) lets large GEMMs hit tensor cores automatically. We have now exposed this via `train.enable_tf32` in `default.yaml` alongside a helper that sets cuDNN and matmul precision to `'highest'`.

**Allocator Tweaks**  
PyTorch supports environment-level control of the caching allocator (`PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync,…`) and pluggable allocators for NCCL or custom flows.[1] After CUDA Graph integration, we should consider enabling `cudaMallocAsync` or the experimental expandable segments if we observe fragmentation during dynamic batch adjustments.

**Action Items**
- Keep `enable_tf32: true` in the training config (already committed).
- Monitor `torch.cuda.memory_summary()` during long runs; experiment with `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` if we hit fragmentation.
- For experimental NCCL allocators, wrap them via `torch.cuda.memory.CUDAPluggableAllocator` once the communication profile demands it.

## 5. Phased Execution Plan

1. **Phase 1 – CUDA Graph Pilot**
   - Refactor the training loop for capture, enable TF32, and integrate graph replay behind a config flag.
   - Validate on synthetic data, then run a full training session comparing throughput.
2. **Phase 2 – CuPy Kernel Prototypes**
   - Identify top hotspots (HRM gate, halting regularizer) with Nsight.
   - Replace with CuPy kernels and benchmark end-to-end impact.
   - Promote stable kernels to PyTorch extensions if needed.
3. **Phase 3 – CUTLASS Expert Kernels**
   - Evaluate MoE expert matmuls; prototype CUTLASS kernels and integrate via extension when they beat ATen.
4. **Phase 4 – Memory/Allocator QA**
   - Stress test long training sessions; tune allocator backend or pluggable allocators if fragmentation or NCCL contention arises.

## References

1. *PyTorch 2.8 CUDA semantics – CUDA Graphs & allocator notes.* https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs
2. *“Accelerating PyTorch with CUDA Graphs,”* PyTorch/NVIDIA Blog (Nov 2024). https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
3. *CuPy User Guide – PyTorch Interoperability.* https://docs.cupy.dev/en/stable/user_guide/interoperability.html
4. *CuPy Installation Guide.* https://docs.cupy.dev/en/stable/install.html
5. *CUTLASS Python README.* https://github.com/NVIDIA/cutlass/blob/main/python/README.md
6. *PyTorch 2.8.0 Release Notes.* https://github.com/pytorch/pytorch/releases/tag/v2.8.0

## 6. CUDA 13.x Support Status (Research Summary)

As of PyTorch **2.8.0** (latest stable release, August 2025), official binaries are published only for CUDA 11.x and CUDA 12.x toolkits. The 2.8 release notes discuss CUDA 12.8/12.9 builds and even call out a Windows-specific workaround for CUDA 12.9.1, but **no CUDA 13 wheels are offered** in the primary “Get Started” selector or release artifacts.[6]

NVIDIA/PyTorch maintainers are actively preparing CUDA 13 coverage, but the work is still in-flight:

- Pull request **[#161663](https://github.com/pytorch/pytorch/pull/161663)** (“Add CUDA 13.0 Windows build”) landed in the main branch to enable Windows CI builds for CUDA 13, but this targets future releases and is not part of 2.8.0 stable artifacts.
- Issue **[#163983](https://github.com/pytorch/pytorch/issues/163983)** (“Add more CI tests for CUDA 13.0”) remains open, noting that CUDA 13 runs only in periodic CI and requesting expanded coverage for PyTorch 2.10.

**Implication for RTX 5090 (Blackwell):** the GPU hardware supports CUDA 13 features, but until PyTorch publishes official CUDA 13 wheels, we should stick with the CUDA 12.9 packages (e.g., the `+cu129` builds) or build PyTorch from source against the CUDA 13 toolkit if we want to experiment. Once the CI work above ships in an official release, revisit upgrading to leverage any CUDA 13-only kernels or driver optimizations.

