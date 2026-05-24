/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Parallel Prefix Sum (Exclusive Scan) — Blelloch algorithm.
//!
//! Demonstrates:
//! - Up-sweep (reduce) phase in shared memory
//! - Root zeroing and down-sweep (distribute) phase
//! - Two-phase barrier-synchronized algorithm
//!
//! The output satisfies: `output[i] = sum_{j < i} input[j]`.
//!
//! Build and run with:
//!   cargo oxide run --manifest-path 02-ported-cuda-samples/02c-parallel-prefix-sum/Cargo.toml

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};

const BLOCK_SIZE: usize = 256;

#[cuda_module]
mod kernels {
    use super::*;

    /// Blelloch exclusive scan (prefix sum) for one block of `BLOCK_SIZE` elements.
    ///
    /// Algorithm:
    /// 1. Load data into shared memory.
    /// 2. **Up-sweep**: build a reduction tree so that `TEMP[idx]` holds the
    ///    sum of its subtree.
    /// 3. Zero the last element (exclusive scan root).
    /// 4. **Down-sweep**: traverse back down, propagating partial sums so
    ///    each position receives the sum of all elements to its left.
    #[kernel]
    pub fn blelloch_scan(input: &[f32], mut output: DisjointSlice<f32>) {
        static mut TEMP: SharedArray<f32, BLOCK_SIZE> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let gid = thread::index_1d().get() as usize;

        // Load from global to shared memory.
        unsafe {
            TEMP[tid] = if gid < input.len() { input[gid] } else { 0.0 };
        }
        thread::sync_threads();

        // ---------- Up-sweep (reduce) ----------
        let mut stride = 1usize;
        while stride < BLOCK_SIZE {
            thread::sync_threads();
            let idx = (tid + 1) * stride * 2 - 1;
            if idx < BLOCK_SIZE {
                unsafe {
                    TEMP[idx] += TEMP[idx - stride];
                }
            }
            stride *= 2;
        }

        // Root of the tree holds the total sum; zero it for exclusive scan.
        if tid == 0 {
            unsafe {
                TEMP[BLOCK_SIZE - 1] = 0.0;
            }
        }
        thread::sync_threads();

        // ---------- Down-sweep (distribute) ----------
        stride = BLOCK_SIZE / 2;
        while stride > 0 {
            thread::sync_threads();
            let idx = (tid + 1) * stride * 2 - 1;
            if idx < BLOCK_SIZE {
                unsafe {
                    let left = TEMP[idx - stride];
                    TEMP[idx - stride] = TEMP[idx];
                    TEMP[idx] += left;
                }
            }
            stride /= 2;
        }
        thread::sync_threads();

        // Write results back to global memory.
        if gid < output.len() {
            unsafe {
                if let Some(out_elem) = output.get_mut(thread::index_1d()) {
                    *out_elem = TEMP[tid];
                }
            }
        }
    }
}

fn main() {
    println!("=== Parallel Prefix Sum (Blelloch Exclusive Scan) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load kernel module");

    const N: usize = BLOCK_SIZE;
    let input: Vec<f32> = (1..=N).map(|i| i as f32).collect();

    println!("Input size:  {} elements", N);
    println!("Input[0..8]: {:?}\n", &input[..8]);

    let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (BLOCK_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    module
        .blelloch_scan(&stream, cfg, &input_dev, &mut out_dev)
        .expect("Kernel launch failed");

    let result = out_dev.to_host_vec(&stream).unwrap();

    println!("Output[0..8]: {:?}", &result[..8]);
    println!("Output[248..256]: {:?}\n", &result[248..]);

    // Verify: output[i] should be the sum of all input[j] for j < i.
    let mut expected = 0.0f32;
    let mut ok = true;
    for i in 0..N {
        if (result[i] - expected).abs() > 1e-3 {
            if ok {
                eprintln!(
                    "  Mismatch at [{}]: expected {}, got {}",
                    i, expected, result[i]
                );
            }
            ok = false;
        }
        expected += input[i];
    }

    // The last element should be the sum of all elements except the last input.
    let total_sum: f32 = input.iter().sum();
    let last_expected = total_sum - input[N - 1];
    println!(
        "Last element:  got = {:.1}, expected = {:.1}  {}",
        result[N - 1],
        last_expected,
        if (result[N - 1] - last_expected).abs() < 1e-3 {
            "✓"
        } else {
            "✗"
        }
    );

    println!(
        "\n{}  {}",
        if ok { "✓ PASS" } else { "✗ FAIL" },
        if ok {
            "Exclusive scan verified: output[i] == sum(input[0..i])"
        } else {
            "Some elements incorrect."
        }
    );

    if !ok {
        std::process::exit(1);
    }
}
