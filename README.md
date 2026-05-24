# cuda-oxide Demos

Standalone demo projects showcasing [cuda-oxide](https://github.com/NVlabs/cuda-oxide) —
a Rust-to-CUDA compiler that compiles single-source Rust to both PTX (GPU) and
native host code.

## Prerequisites

- **cuda-oxide toolchain**: Install `cargo-oxide` and the required Rust nightly
  toolchain. See the [cuda-oxide README](https://github.com/NVlabs/cuda-oxide) for setup.
- **NVIDIA GPU**: sm_80+ (Ampere or newer) recommended.

## Building & Running

All demos are workspace members. Build and run with:

```bash
# Build everything
cargo build --workspace

# Run a specific demo
cargo oxide run --manifest-path 01-hello-world/01a-basic-vecadd/Cargo.toml

# Or from the workspace root (if cargo-oxide supports workspace member paths)
cargo oxide run -p basic-vecadd
```

## Demo Categories

### 01 — Hello World (Onboarding)

| Demo | Description | Key Concepts |
|------|-------------|-------------|
| `01a-basic-vecadd` | Vector addition | Kernel launch, DeviceBuffer, verification |
| `01b-closure-params` | Closure arguments | Host closures captured by kernel |
| `01c-generic-kernel` | Generic kernel | Monomorphization for f32/i32 |
| `01d-cross-crate` | Cross-crate kernel | Library + binary separation |

### 02 — Ported CUDA Samples

| Demo | Description | Key Concepts |
|------|-------------|-------------|
| `02a-matrix-transpose` | Matrix transpose | Shared memory tiling, 2D indexing |
| `02b-reduction` | Parallel reduction | Tree reduction, warp shuffle |
| `02c-parallel-prefix-sum` | Prefix sum | Blelloch scan, shared memory |
| `02d-convolution-2d` | 2D convolution | Stencil, boundary handling |
| `02e-nbody` | N-body simulation | O(n²) pairwise forces, shared memory tiling |

### 03 — ML Inference

| Demo | Description | Key Concepts |
|------|-------------|-------------|
| `03a-mnist-mlp` | MLP forward pass | FC → ReLU → Softmax |
| `03b-transformer-attention` | Self-attention | Q·Kᵀ, softmax, attention weights |
| `03c-conv2d-im2col` | Conv2D via im2col | im2col + GEMM pattern |
| `03d-llm-kv-cache` | LLM KV cache | Async pipeline, cache management |

### 04 — Scientific Computing

| Demo | Description | Key Concepts |
|------|-------------|-------------|
| `04a-heat-diffusion` | 2D heat equation | Stencil, double-buffering |
| `04b-wave-equation` | 2D wave equation | Finite differences, time stepping |
| `04c-monte-carlo-pi` | Pi estimation | Monte Carlo, atomics, RNG |
| `04d-spmv` | Sparse mat-vec multiply | CSR format, indirect indexing |
| `04e-conjugate-gradient` | CG solver | Iterative solver, dot products |

### 05 — GPU Data Structures

| Demo | Description | Key Concepts |
|------|-------------|-------------|
| `05a-prefix-tree` | Prefix tree | Parallel tree construction |
| `05b-bvh-raytracer` | BVH ray tracing | Hierarchy traversal |
| `05c-graph-traversal` | Graph BFS | Frontier expansion, atomics |
| `05d-sorting-networks` | Bitonic sort | Compare-and-swap networks |

### 06 — Multi-GPU

| Demo | Description | Key Concepts |
|------|-------------|-------------|
| `06a-peer-access` | P2P memory | Inter-GPU communication |
| `06b-nccl-wrapper` | NCCL allreduce | Multi-GPU collectives |
| `06c-pipeline-parallel` | Pipeline parallelism | Async across GPUs |

### 07 — Rust Ecosystem Integration

| Demo | Description | Integrates With |
|------|-------------|-----------------|
| `07a-ndarray-bridge` | Zero-copy ndarray | `ndarray` crate |
| `07b-image-processing` | GPU image filters | `image` crate |
| `07c-random-numbers` | Parallel RNG | `rand` crate |
| `07d-serde-kernels` | Serialized configs | `serde` crate |

## Notes

- Demos use **path dependencies** pointing to a sibling `cuda-oxide/` directory.
  To use published versions instead, edit `Cargo.toml` to replace the paths with
  `git = "https://github.com/NVlabs/cuda-oxide.git"`.
- Some demos (06-*, Hopper features) require multiple GPUs or specific hardware.
  Check the doc comment at the top of each `main.rs` for requirements.
