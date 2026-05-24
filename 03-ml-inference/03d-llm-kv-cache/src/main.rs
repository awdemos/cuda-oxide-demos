/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! LLM KV Cache — autoregressive inference with cached key/value pairs
//!
//! Simulates a simplified LLM decoding pipeline:
//!
//!   1. Append K and V vectors for each token to the KV cache.
//!   2. Run attention over the full cached sequence using a query vector.
//!
//! Build and run with:
//!   cargo oxide run --manifest-path 03-ml-inference/03d-llm-kv-cache/Cargo.toml

#![allow(clippy::needless_range_loop)]

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};

// =============================================================================
// KERNELS
// =============================================================================

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn cache_append(
        key: &[f32],
        val: &[f32],
        mut cache_k: DisjointSlice<f32>,
        mut cache_v: DisjointSlice<f32>,
        pos: u32,
        d_model: u32,
    ) {
        let i = thread::index_1d().get() as usize;
        if i >= d_model as usize {
            return;
        }
        let cache_idx = pos as usize * d_model as usize + i;
        unsafe {
            *cache_k.get_unchecked_mut(cache_idx) = key[i];
            *cache_v.get_unchecked_mut(cache_idx) = val[i];
        }
    }

    #[kernel]
    pub fn attend(
        q: &[f32],
        cache_k: &[f32],
        cache_v: &[f32],
        mut out: DisjointSlice<f32>,
        seq_len: u32,
        d_model: u32,
    ) {
        let i = thread::index_1d().get() as usize;
        let d = d_model as usize;
        if i >= d {
            return;
        }

        let seq = seq_len as usize;

        let mut max_score = -1e10f32;
        for pos in 0..seq {
            let mut score = 0.0f32;
            for j in 0..d {
                score += q[j] * cache_k[pos * d + j];
            }
            score /= (d as f32).sqrt();
            if score > max_score {
                max_score = score;
            }
        }

        let mut sum_exp = 0.0f32;
        for pos in 0..seq {
            let mut score = 0.0f32;
            for j in 0..d {
                score += q[j] * cache_k[pos * d + j];
            }
            score /= (d as f32).sqrt();
            sum_exp += (score - max_score).exp();
        }

        let mut result = 0.0f32;
        for pos in 0..seq {
            let mut score = 0.0f32;
            for j in 0..d {
                score += q[j] * cache_k[pos * d + j];
            }
            score /= (d as f32).sqrt();
            let attn = (score - max_score).exp() / sum_exp;
            result += attn * cache_v[pos * d + i];
        }

        if let Some(e) = out.get_mut(thread::index_1d()) {
            *e = result;
        }
    }
}

// =============================================================================
// HOST HELPERS
// =============================================================================

fn host_attend(
    q: &[f32],
    cache_k: &[f32],
    cache_v: &[f32],
    seq_len: usize,
    d_model: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; d_model];

    let mut max_score = -1e10f32;
    for pos in 0..seq_len {
        let mut score = 0.0f32;
        for j in 0..d_model {
            score += q[j] * cache_k[pos * d_model + j];
        }
        score /= (d_model as f32).sqrt();
        if score > max_score {
            max_score = score;
        }
    }

    let mut sum_exp = 0.0f32;
    for pos in 0..seq_len {
        let mut score = 0.0f32;
        for j in 0..d_model {
            score += q[j] * cache_k[pos * d_model + j];
        }
        score /= (d_model as f32).sqrt();
        sum_exp += (score - max_score).exp();
    }

    for i in 0..d_model {
        let mut result = 0.0f32;
        for pos in 0..seq_len {
            let mut score = 0.0f32;
            for j in 0..d_model {
                score += q[j] * cache_k[pos * d_model + j];
            }
            score /= (d_model as f32).sqrt();
            let attn = (score - max_score).exp() / sum_exp;
            result += attn * cache_v[pos * d_model + i];
        }
        out[i] = result;
    }

    out
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() {
    println!("=== LLM KV Cache ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load kernel module");

    const SEQ_LEN: usize = 4;
    const D_MODEL: usize = 8;

    let tokens_k: Vec<Vec<f32>> = (0..SEQ_LEN)
        .map(|t| {
            (0..D_MODEL)
                .map(|i| ((t * D_MODEL + i) as f32 * 0.1).sin())
                .collect()
        })
        .collect();

    let tokens_v: Vec<Vec<f32>> = (0..SEQ_LEN)
        .map(|t| {
            (0..D_MODEL)
                .map(|i| ((t * D_MODEL + i) as f32 * 0.1).cos())
                .collect()
        })
        .collect();

    let mut cache_k_dev = DeviceBuffer::<f32>::zeroed(&stream, SEQ_LEN * D_MODEL).unwrap();
    let mut cache_v_dev = DeviceBuffer::<f32>::zeroed(&stream, SEQ_LEN * D_MODEL).unwrap();

    let cache_cfg = LaunchConfig::for_num_elems(D_MODEL as u32);

    println!("Appending {} tokens to KV cache...", SEQ_LEN);
    for pos in 0..SEQ_LEN {
        let key_dev = DeviceBuffer::from_host(&stream, &tokens_k[pos]).unwrap();
        let val_dev = DeviceBuffer::from_host(&stream, &tokens_v[pos]).unwrap();

        module
            .cache_append(
                &stream,
                cache_cfg,
                &key_dev,
                &val_dev,
                &mut cache_k_dev,
                &mut cache_v_dev,
                pos as u32,
                D_MODEL as u32,
            )
            .expect("cache_append failed");

        println!("  Token {} → cache appended", pos);
    }

    let q: Vec<f32> = (0..D_MODEL).map(|i| ((i as f32) * 0.15).sin()).collect();
    let q_dev = DeviceBuffer::from_host(&stream, &q).unwrap();

    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, D_MODEL).unwrap();
    let attn_cfg = LaunchConfig::for_num_elems(D_MODEL as u32);

    module
        .attend(
            &stream,
            attn_cfg,
            &q_dev,
            &cache_k_dev,
            &cache_v_dev,
            &mut out_dev,
            SEQ_LEN as u32,
            D_MODEL as u32,
        )
        .expect("attend failed");

    let result = out_dev.to_host_vec(&stream).unwrap();
    let cache_k: Vec<f32> = cache_k_dev.to_host_vec(&stream).unwrap();
    let cache_v: Vec<f32> = cache_v_dev.to_host_vec(&stream).unwrap();
    let expected = host_attend(&q, &cache_k, &cache_v, SEQ_LEN, D_MODEL);

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

    println!("\nAttention output: {:?}", &result[..]);
    println!("Max error vs host reference: {:.6e}", max_error);
    println!("Mismatch count: {}/{}", errors, expected.len());

    if max_error < 1e-3 {
        println!("\n✓ PASS: LLM KV cache attention completed successfully.");
    } else {
        println!("\n✗ FAIL: error too large.");
        std::process::exit(1);
    }
}
