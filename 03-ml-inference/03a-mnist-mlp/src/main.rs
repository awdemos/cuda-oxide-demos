/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! MNIST MLP — 3-layer fully-connected forward pass (784→128→64→10)
//!
//! Demonstrates a complete MLP inference pipeline with:
//! - `linear_fwd` kernel: one thread per output neuron computes y = W·x + b
//! - `relu` kernel: element-wise activation
//!
//! Build and run with:
//!   cargo oxide run --manifest-path 03-ml-inference/03a-mnist-mlp/Cargo.toml

#![allow(clippy::needless_range_loop)]

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};

// =============================================================================
// KERNELS
// =============================================================================

#[cuda_module]
mod kernels {
    use super::*;

    /// Fully-connected forward pass: out[i] = sum_j(in[j] * W[j·out_dim + i]) + b[i]
    ///
    /// One thread per output element. Weight matrix is stored in row-major
    /// layout with shape [in_dim, out_dim] so that W[j][i] = weight[j·out_dim+i].
    #[kernel]
    pub fn linear_fwd(
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
        mut out: DisjointSlice<f32>,
        in_dim: u32,
        out_dim: u32,
    ) {
        let i = thread::index_1d().get() as usize;
        if i >= out_dim as usize {
            return;
        }
        let mut sum = bias[i];
        let out_d = out_dim as usize;
        for j in 0..in_dim as usize {
            sum += input[j] * weight[j * out_d + i];
        }
        if let Some(e) = out.get_mut(thread::index_1d()) {
            *e = sum;
        }
    }

    /// ReLU activation: out[i] = max(0, inp[i])
    #[kernel]
    pub fn relu(inp: &[f32], mut out: DisjointSlice<f32>, n: u32) {
        let i = thread::index_1d().get() as usize;
        if i >= n as usize {
            return;
        }
        let v = inp[i];
        if let Some(e) = out.get_mut(thread::index_1d()) {
            *e = if v > 0.0f32 { v } else { 0.0f32 };
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn host_linear_fwd(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let mut sum = bias[i];
        for j in 0..in_dim {
            sum += input[j] * weight[j * out_dim + i];
        }
        out[i] = sum;
    }
    out
}

fn host_relu(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|&v| if v > 0.0 { v } else { 0.0 })
        .collect()
}

fn main() {
    println!("=== 3-Layer MLP Forward Pass (784→128→64→10) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load kernel module");

    const DIMS: &[(u32, u32)] = &[(784, 128), (128, 64), (64, 10)];

    let mut x: Vec<f32> = (0..784).map(|i| (i as f32) / 784.0).collect();
    let mut all_pass = true;

    for (layer, &(in_dim, out_dim)) in DIMS.iter().enumerate() {
        let w: Vec<f32> = (0..in_dim * out_dim)
            .map(|i| (((i * 17 + 31) % 100) as f32 - 50.0) * 0.002)
            .collect();
        let b: Vec<f32> = (0..out_dim)
            .map(|i| (((i * 13 + 7) % 20) as f32 - 10.0) * 0.01)
            .collect();

        let x_in = x.clone();
        let x_dev = DeviceBuffer::from_host(&stream, &x_in).unwrap();
        let w_dev = DeviceBuffer::from_host(&stream, &w).unwrap();
        let b_dev = DeviceBuffer::from_host(&stream, &b).unwrap();
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, out_dim as usize).unwrap();

        let cfg = LaunchConfig::for_num_elems(out_dim);
        module
            .linear_fwd(
                &stream,
                cfg,
                &x_dev,
                &w_dev,
                &b_dev,
                &mut out_dev,
                in_dim,
                out_dim,
            )
            .expect("linear_fwd failed");

        x = out_dev.to_host_vec(&stream).unwrap();

        let host_out = host_linear_fwd(&x_in, &w, &b, in_dim as usize, out_dim as usize);
        let mut layer_max_err = 0.0f32;
        for i in 0..out_dim as usize {
            let err = (x[i] - host_out[i]).abs();
            if err > layer_max_err {
                layer_max_err = err;
            }
        }
        if layer_max_err > 1e-3 {
            all_pass = false;
            println!(
                "  Layer {} linear max error = {:.6e} ✗",
                layer + 1,
                layer_max_err
            );
        }

        if layer < 2 {
            let x_dev = DeviceBuffer::from_host(&stream, &x).unwrap();
            let mut act_dev = DeviceBuffer::<f32>::zeroed(&stream, out_dim as usize).unwrap();
            module
                .relu(
                    &stream,
                    LaunchConfig::for_num_elems(out_dim),
                    &x_dev,
                    &mut act_dev,
                    out_dim,
                )
                .expect("relu failed");
            x = act_dev.to_host_vec(&stream).unwrap();

            let host_relu_out = host_relu(&host_out);
            let mut relu_max_err = 0.0f32;
            for i in 0..out_dim as usize {
                let err = (x[i] - host_relu_out[i]).abs();
                if err > relu_max_err {
                    relu_max_err = err;
                }
            }
            if relu_max_err > 1e-4 {
                all_pass = false;
                println!(
                    "  Layer {} ReLU max error = {:.6e} ✗",
                    layer + 1,
                    relu_max_err
                );
            }
        }

        println!(
            "Layer {} ({}→{}): first 5 outputs = {:?}",
            layer + 1,
            in_dim,
            out_dim,
            &x[..5.min(x.len())]
        );
    }

    println!("\nOutput layer activations (all 10): {:?}", &x[..]);

    if all_pass {
        println!("\n✓ PASS: MLP forward pass completed successfully.");
    } else {
        println!("\n✗ FAIL: verification errors detected.");
        std::process::exit(1);
    }
}
