# Agent Notes: cuda-oxide-demos

Standalone Rust workspace demonstrating [cuda-oxide](https://github.com/NVlabs/cuda-oxide), a single-source Rust-to-CUDA compiler that emits PTX for the GPU and native host code.

## Repository Layout

This is a Cargo workspace with one demo per crate under numbered categories:

- `01-hello-world/` — Onboarding: vector addition, closures, generics, cross-crate kernels.
- `02-ported-cuda-samples/` — Shared memory, reduction, prefix sum, convolution, N-body.
- `03-ml-inference/` — MNIST MLP, transformer attention, im2col convolution, LLM KV cache.
- `04-scientific-computing/` — Heat diffusion, wave equation, Monte Carlo π, SpMV, conjugate gradient.
- `05-data-structures/` — Prefix tree, BVH raytracer, graph traversal, sorting networks.
- `06-multi-gpu/` — Peer access, NCCL wrapper, pipeline parallelism.
- `07-ecosystem-integration/` — ndarray bridge, image processing, random numbers, serde kernels.

Common files:

- `Cargo.toml` — Workspace manifest, edition 2024.
- `.cargo/config.toml` — `cargo-oxide` / nightly toolchain configuration.
- `README.md` — Per-category and per-demo instructions.

## Prerequisites

- The `cuda-oxide` toolchain and `cargo-oxide` plugin.
- A recent Rust nightly toolchain (see the upstream README for the exact version).
- NVIDIA GPU with sm_80+ (Ampere or newer) recommended.
- NVIDIA driver and CUDA toolkit installed.

## Build

```bash
# Build all host code
cargo build --workspace

# Build and run a specific demo with cargo-oxide
cargo oxide run --manifest-path 01-hello-world/01a-basic-vecadd/Cargo.toml
```

Some demos can also be invoked by package name from the workspace root if `cargo-oxide` supports workspace member paths:

```bash
cargo oxide run -p basic-vecadd
```

## Test

There is no top-level test suite; each demo verifies itself by checking device results against host references. A demo run that exits successfully is a passing check.

```bash
# Run a quick sanity check on the first demo
cargo oxide run --manifest-path 01-hello-world/01a-basic-vecadd/Cargo.toml
```

## Key Conventions

- Each demo is self-contained. Read its local `README.md` for specific inputs/outputs and expected TFLOPS/speedups.
- `.cargo/config.toml` may pin a nightly channel; do not override it with a stable compiler.
- Generated IR/PTX artifacts are typically ignored by `.gitignore`; do not commit them.

## Common Issues

- **"cargo-oxide not found"**: install the cuda-oxide toolchain first per upstream instructions.
- **PTX compile error on older GPU**: some kernels require sm_80+; lower the arch target if the demo supports it.
- **Nightly toolchain missing**: use `rustup toolchain install nightly-YYYY-MM-DD` as specified by cuda-oxide.

## License

Apache-2.0. See `Cargo.toml`.
