/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Transformer Attention — scaled dot-product attention with softmax
//!
//! Implements single-head self-attention in one kernel:
//!
//!   Attention(Q, K, V) = softmax(Q · K^T / sqrt(d_k)) · V
//!
//! Each thread computes one output element (row, col) where row is the
//! sequence position and col is the dimension within the head.
//!
//! Dimensions: seq_len = 8, d_k = 16.
//!
//! Build and run with:
//!   cargo oxide run --manifest-path 03-ml-inference/03b-transformer-attention/Cargo.toml

#![allow(clippy::needless_range_loop)]

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};

// =============================================================================
// KERNELS
// =============================================================================

#[cuda_module]
mod kernels {
    use super::*;

    /// Scaled dot-product attention.
    ///
    /// Each thread computes one output element out[row * d_k + col].
    /// Threads redundantly compute the softmax for their row.
    #[kernel]
    pub fn attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        mut out: DisjointSlice<f32>,
        N: u32,
        d_k: u32,
    ) {
        let idx = thread::index_1d().get() as usize;
        let n = N as usize;
        let d = d_k as usize;
        let total = n * d;
        if idx >= total {
            return;
        }

        let row = idx / d;
        let col = idx % d;

        let mut max_score = -1e10f32;
        for pos in 0..n {
            let mut score = 0.0f32;
            for t in 0..d {
                score += q[row * d + t] * k[pos * d + t];
            }
            score /= (d as f32).sqrt();
            if score > max_score {
                max_score = score;
            }
        }

        let mut sum_exp = 0.0f32;
        for pos in 0..n {
            let mut score = 0.0f32;
            for t in 0..d {
                score += q[row * d + t] * k[pos * d + t];
            }
            score /= (d as f32).sqrt();
            sum_exp += (score - max_score).exp();
        }

        let mut result = 0.0f32;
        for pos in 0..n {
            let mut score = 0.0f32;
            for t in 0..d {
                score += q[row * d + t] * k[pos * d + t];
            }
            score /= (d as f32).sqrt();
            let attn = (score - max_score).exp() / sum_exp;
            result += attn * v[pos * d + col];
        }

        if let Some(e) = out.get_mut(thread::index_1d()) {
            *e = result;
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn host_attention(q: &[f32], k: &[f32], v: &[f32], n: usize, d_k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * d_k];
    for row in 0..n {
        let mut scores = vec![0.0f32; n];
        let mut max_score = -1e10f32;
        for pos in 0..n {
            let mut score = 0.0f32;
            for t in 0..d_k {
                score += q[row * d_k + t] * k[pos * d_k + t];
            }
            score /= (d_k as f32).sqrt();
            scores[pos] = score;
            if score > max_score {
                max_score = score;
            }
        }

        let mut sum_exp = 0.0f32;
        for pos in 0..n {
            sum_exp += (scores[pos] - max_score).exp();
        }

        for col in 0..d_k {
            let mut result = 0.0f32;
            for pos in 0..n {
                let attn = (scores[pos] - max_score).exp() / sum_exp;
                result += attn * v[pos * d_k + col];
            }
            out[row * d_k + col] = result;
        }
    }
    out
}

fn main() {
    println!("=== Scaled Dot-Product Attention ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load kernel module");

    const N: usize = 8;
    const D_K: usize = 16;

    let q: Vec<f32> = (0..N * D_K).map(|i| ((i as f32) * 0.1).sin()).collect();
    let k: Vec<f32> = (0..N * D_K).map(|i| ((i as f32) * 0.13).cos()).collect();
    let v: Vec<f32> = (0..N * D_K)
        .map(|i| ((i as f32) * 0.07).sin() + 0.5)
        .collect();

    let q_dev = DeviceBuffer::from_host(&stream, &q).unwrap();
    let k_dev = DeviceBuffer::from_host(&stream, &k).unwrap();
    let v_dev = DeviceBuffer::from_host(&stream, &v).unwrap();

    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N * D_K).unwrap();

    let cfg = LaunchConfig::for_num_elems((N * D_K) as u32);
    module
        .attention(
            &stream,
            cfg,
            &q_dev,
            &k_dev,
            &v_dev,
            &mut out_dev,
            N as u32,
            D_K as u32,
        )
        .expect("attention kernel failed");

    let result = out_dev.to_host_vec(&stream).unwrap();

    let expected = host_attention(&q, &k, &v, N, D_K);

    let mut max_error = 0.0f32;
    let mut errors = 0usize;
    for i in 0..expected.len() {
        let err = (result[i] - expected[i]).abs();
        if err > max_error {
            max_error = err;
        }
        if err > 1e-3 {
            errors += 1;
            if errors <= 5 {
                println!(
                    "  Mismatch at [{}]: expected {:.6}, got {:.6}",
                    i, expected[i], result[i]
                );
            }
        }
    }

    println!(
        "Attention output [0..4]: {:?}",
        &result[..4.min(result.len())]
    );
    println!("Max error vs host reference: {:.6e}", max_error);
    println!("Mismatch count: {}/{}", errors, expected.len());

    if max_error < 1e-3 {
        println!("\n✓ PASS: Attention forward pass completed successfully.");
    } else {
        println!("\n✗ FAIL: error too large.");
        std::process::exit(1);
    }
}
