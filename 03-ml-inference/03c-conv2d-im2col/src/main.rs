/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Conv2D via im2col + GEMM
//!
//! Demonstrates the classic im2col convolution decomposition:
//!
//!   1. Pad the input with zeros (same padding, 3×3 kernel, stride 1).
//!   2. `im2col` kernel: extract K×K patches into a column matrix.
//!   3. `sgemm` kernel: multiply weight matrix [F, C·K·K] by im2col [C·K·K, H·W].
//!   4. Result is the convolution output [F, H·W].
//!
//! Verified against a host-side reference implementation.
//!
//! Build and run with:
//!   cargo oxide run --manifest-path 03-ml-inference/03c-conv2d-im2col/Cargo.toml

#![allow(clippy::needless_range_loop)]

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};

// =============================================================================
// KERNELS
// =============================================================================

#[cuda_module]
mod kernels {
    use super::*;

    /// im2col: extract K×K patches from padded input into column matrix.
    ///
    /// Output layout: rows = C·K·K, cols = H·W
    /// Each thread writes one element of the im2col matrix.
    #[kernel]
    pub fn im2col(input: &[f32], mut cols: DisjointSlice<f32>, C: u32, H: u32, W: u32, K: u32) {
        let idx = thread::index_1d().get() as usize;
        let out_h = H as usize;
        let out_w = W as usize;
        let total_cols = out_h * out_w;
        let total_rows = C as usize * K as usize * K as usize;
        let total_elems = total_rows * total_cols;

        if idx >= total_elems {
            return;
        }

        let filter_idx = idx / total_cols;
        let patch_idx = idx % total_cols;

        let c = filter_idx / (K as usize * K as usize);
        let k_rem = filter_idx % (K as usize * K as usize);
        let ky = k_rem / K as usize;
        let kx = k_rem % K as usize;

        let h = patch_idx / out_w;
        let w = patch_idx % out_w;

        let in_idx =
            c * (H + 2) as usize * (W + 2) as usize + (h + ky) * (W + 2) as usize + (w + kx);

        if let Some(e) = cols.get_mut(thread::index_1d()) {
            *e = input[in_idx];
        }
    }

    /// Naive GEMM using 1-D thread indexing.
    ///
    /// C[M][N] = A[M][K] × B[K][N]
    /// Each thread computes one element of C.
    #[kernel]
    pub fn sgemm(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>, M: u32, N: u32, K: u32) {
        let idx = thread::index_1d().get() as usize;
        let total = M as usize * N as usize;
        if idx >= total {
            return;
        }

        let row = idx / N as usize;
        let col = idx % N as usize;

        let mut sum = 0.0f32;
        for k in 0..K as usize {
            sum += a[row * K as usize + k] * b[k * N as usize + col];
        }

        if let Some(e) = c.get_mut(thread::index_1d()) {
            *e = sum;
        }
    }
}

// =============================================================================
// HOST HELPERS
// =============================================================================

/// Host reference Conv2D with same padding.
fn host_conv2d(
    input: &[f32],
    weight: &[f32],
    c: usize,
    h: usize,
    w: usize,
    f: usize,
    k: usize,
) -> Vec<f32> {
    let pad = k / 2;
    let mut output = vec![0.0f32; f * h * w];

    for fi in 0..f {
        for hi in 0..h {
            for wi in 0..w {
                let mut sum = 0.0f32;
                for ci in 0..c {
                    for ky in 0..k {
                        for kx in 0..k {
                            let in_h = hi as i32 + ky as i32 - pad as i32;
                            let in_w = wi as i32 + kx as i32 - pad as i32;
                            let in_val =
                                if in_h >= 0 && in_h < h as i32 && in_w >= 0 && in_w < w as i32 {
                                    input[ci * h * w + (in_h as usize) * w + (in_w as usize)]
                                } else {
                                    0.0f32
                                };
                            let w_val = weight[fi * c * k * k + ci * k * k + ky * k + kx];
                            sum += in_val * w_val;
                        }
                    }
                }
                output[fi * h * w + hi * w + wi] = sum;
            }
        }
    }
    output
}

/// Pad input with zeros on all sides (same padding).
fn pad_input(input: &[f32], c: usize, h: usize, w: usize, pad: usize) -> Vec<f32> {
    let h_p = h + 2 * pad;
    let w_p = w + 2 * pad;
    let mut padded = vec![0.0f32; c * h_p * w_p];
    for ci in 0..c {
        for hi in 0..h {
            for wi in 0..w {
                padded[ci * h_p * w_p + (hi + pad) * w_p + (wi + pad)] =
                    input[ci * h * w + hi * w + wi];
            }
        }
    }
    padded
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() {
    println!("=== Conv2D via im2col + GEMM ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load kernel module");

    const C: usize = 1;
    const H: usize = 8;
    const W: usize = 8;
    const F: usize = 1;
    const K: usize = 3;
    const PAD: usize = K / 2;

    let input: Vec<f32> = (0..C * H * W)
        .map(|i| {
            let row = i / W;
            let col = i % W;
            (row + col) as f32 / 14.0
        })
        .collect();

    let weight: Vec<f32> = vec![1.0f32; F * C * K * K];

    println!("Input shape:  {}×{}×{}×{}", 1, C, H, W);
    println!("Filter shape: {}×{}×{}×{}", F, C, K, K);
    println!("Output shape: {}×{}×{}×{}\n", 1, F, H, W);

    let padded_input = pad_input(&input, C, H, W, PAD);
    let padded_dev = DeviceBuffer::from_host(&stream, &padded_input).unwrap();

    let im2col_rows = C * K * K;
    let im2col_cols = H * W;
    let im2col_size = im2col_rows * im2col_cols;
    let mut cols_dev = DeviceBuffer::<f32>::zeroed(&stream, im2col_size).unwrap();

    let im2col_cfg = LaunchConfig::for_num_elems(im2col_size as u32);
    module
        .im2col(
            &stream,
            im2col_cfg,
            &padded_dev,
            &mut cols_dev,
            C as u32,
            H as u32,
            W as u32,
            K as u32,
        )
        .expect("im2col failed");

    let weight_dev = DeviceBuffer::from_host(&stream, &weight).unwrap();
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, F * H * W).unwrap();

    let gemm_cfg = LaunchConfig::for_num_elems((F * H * W) as u32);
    module
        .sgemm(
            &stream,
            gemm_cfg,
            &weight_dev,
            &cols_dev,
            &mut out_dev,
            F as u32,
            (H * W) as u32,
            (C * K * K) as u32,
        )
        .expect("sgemm failed");

    let result = out_dev.to_host_vec(&stream).unwrap();

    let expected = host_conv2d(&input, &weight, C, H, W, F, K);

    let mut max_error = 0.0f32;
    let mut errors = 0usize;
    for i in 0..expected.len() {
        let err = (result[i] - expected[i]).abs();
        if err > max_error {
            max_error = err;
        }
        if err > 1e-4 {
            errors += 1;
            if errors <= 5 {
                println!(
                    "  Mismatch at [{}]: expected {:.6}, got {:.6}",
                    i, expected[i], result[i]
                );
            }
        }
    }

    println!("Max error vs host reference: {:.6e}", max_error);
    println!("Mismatch count: {}/{}", errors, expected.len());

    println!("\nOutput (filter 0, {}×{}):", H, W);
    for row in 0..H {
        for col in 0..W {
            print!(" {:7.3}", result[row * W + col]);
        }
        println!();
    }

    if max_error < 1e-3 {
        println!("\n✓ PASS: im2col + GEMM convolution completed successfully.");
    } else {
        println!("\n✗ FAIL: max error too large.");
        std::process::exit(1);
    }
}
