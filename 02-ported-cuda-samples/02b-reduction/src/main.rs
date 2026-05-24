/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Parallel Reduction — classic CUDA sample ported to cuda-oxide.
//!
//! Demonstrates three reduction strategies:
//! 1. **Naïve (divergent)** — adjacent pairing with modulo, causing warp divergence
//! 2. **Shared memory (tree)** — contiguous thread pairing, minimal divergence
//! 3. **Warp shuffle** — register-to-register reduction, no shared memory for intra-warp
//!
//! Multi-block: each block produces a partial sum; the host adds the partials.
//!
//! Build and run with:
//!   cargo oxide run --manifest-path 02-ported-cuda-samples/02b-reduction/Cargo.toml

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, warp, DisjointSlice, SharedArray};

const BLOCK_SIZE: usize = 256;

#[cuda_module]
mod kernels {
    use super::*;

    /// Naïve reduction: adjacent pairing with `%` operator.
    ///
    /// In each step only threads whose index is a multiple of `2*stride`
    /// are active, causing severe warp divergence (only 1 thread active
    /// in the final warp-level steps).
    #[kernel]
    pub fn reduce_naive(input: &[f32], mut partial_sums: DisjointSlice<f32>, n: u32) {
        static mut SMEM: SharedArray<f32, BLOCK_SIZE> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let bid = thread::blockIdx_x() as usize;
        let gid = bid * BLOCK_SIZE + tid;
        let n_val = n as usize;

        unsafe {
            SMEM[tid] = if gid < n_val { input[gid] } else { 0.0 };
        }
        thread::sync_threads();

        // Adjacent pairing — causes divergence because active threads are
        // strided (tid % (2*stride) == 0) rather than contiguous.
        let mut stride = 1usize;
        while stride < BLOCK_SIZE {
            thread::sync_threads();
            if tid % (2 * stride) == 0 {
                unsafe {
                    SMEM[tid] += SMEM[tid + stride];
                }
            }
            stride *= 2;
        }

        if tid == 0 {
            unsafe {
                *partial_sums.get_unchecked_mut(bid) = SMEM[0];
            }
        }
    }

    /// Optimized tree reduction using shared memory with contiguous threads.
    ///
    /// Active threads are `tid < stride`, which keeps warps either fully
    /// active or fully inactive — much less divergence than the naïve
    /// modulo-based approach.
    #[kernel]
    pub fn reduce_shared(input: &[f32], mut partial_sums: DisjointSlice<f32>, n: u32) {
        static mut SMEM: SharedArray<f32, BLOCK_SIZE> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let bid = thread::blockIdx_x() as usize;
        let gid = bid * BLOCK_SIZE + tid;
        let n_val = n as usize;

        unsafe {
            SMEM[tid] = if gid < n_val { input[gid] } else { 0.0 };
        }
        thread::sync_threads();

        // Tree reduction: contiguous active threads.
        let mut stride = BLOCK_SIZE / 2;
        while stride > 0 {
            if tid < stride {
                unsafe {
                    SMEM[tid] += SMEM[tid + stride];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }

        if tid == 0 {
            unsafe {
                *partial_sums.get_unchecked_mut(bid) = SMEM[0];
            }
        }
    }

    /// Warp-shuffle reduction.
    ///
    /// Intra-warp sums are computed with `shuffle_down` (register-to-register,
    /// no shared memory, no barrier). Inter-warp sums use a small shared
    /// array (one slot per warp) and a final warp-level reduction.
    #[kernel]
    pub fn reduce_warp_shuffle(input: &[f32], mut partial_sums: DisjointSlice<f32>, n: u32) {
        // One slot per warp (BLOCK_SIZE / 32 = 8 warps).
        static mut SMEM: SharedArray<f32, { BLOCK_SIZE / 32 }> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let bid = thread::blockIdx_x() as usize;
        let gid = bid * BLOCK_SIZE + tid;
        let lane = warp::lane_id() as usize;
        let warp_id = warp::warp_id() as usize;
        let n_val = n as usize;

        let mut val = if gid < n_val { input[gid] } else { 0.0 };

        // Warp-level reduction using shuffle_down.
        val += warp::shuffle_down_f32(val, 16);
        val += warp::shuffle_down_f32(val, 8);
        val += warp::shuffle_down_f32(val, 4);
        val += warp::shuffle_down_f32(val, 2);
        val += warp::shuffle_down_f32(val, 1);

        // Lane 0 of each warp writes its warp sum to shared memory.
        if lane == 0 {
            unsafe {
                SMEM[warp_id] = val;
            }
        }
        thread::sync_threads();

        // Warp 0 reduces the warp sums.
        if warp_id == 0 {
            let mut warp_sum = if lane < (BLOCK_SIZE / 32) {
                unsafe { SMEM[lane] }
            } else {
                0.0
            };
            warp_sum += warp::shuffle_down_f32(warp_sum, 4);
            warp_sum += warp::shuffle_down_f32(warp_sum, 2);
            warp_sum += warp::shuffle_down_f32(warp_sum, 1);
            if lane == 0 {
                unsafe {
                    *partial_sums.get_unchecked_mut(bid) = warp_sum;
                }
            }
        }
    }
}

fn main() {
    println!("=== Parallel Reduction ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load kernel module");

    const N: usize = 1_048_576; // 2^20
    let data: Vec<f32> = (0..N).map(|i| ((i % 100) as f32) * 0.01).collect();
    let expected: f32 = data.iter().sum();

    println!("Input: {} elements", N);
    println!("Expected sum: {:.6}\n", expected);

    let data_dev = DeviceBuffer::from_host(&stream, &data).unwrap();
    let num_blocks = N / BLOCK_SIZE;
    let cfg = LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (BLOCK_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // ----- Naïve (divergent) -----
    println!("--- 1. Naïve (divergent adjacent pairing) ---");
    let mut partial_naive = DeviceBuffer::<f32>::zeroed(&stream, num_blocks).unwrap();
    module
        .reduce_naive(&stream, cfg, &data_dev, &mut partial_naive, N as u32)
        .expect("Naïve kernel launch failed");
    let partial_naive_host = partial_naive.to_host_vec(&stream).unwrap();
    let sum_naive: f32 = partial_naive_host.iter().sum();
    let ok_naive = (sum_naive - expected).abs() < 1.0;
    println!(
        "  Sum = {:.6}  {}  (diff = {:.6})",
        sum_naive,
        if ok_naive { "✓ PASS" } else { "✗ FAIL" },
        (sum_naive - expected).abs()
    );

    // ----- Shared memory (tree) -----
    println!("\n--- 2. Shared Memory (tree, contiguous threads) ---");
    let mut partial_shared = DeviceBuffer::<f32>::zeroed(&stream, num_blocks).unwrap();
    module
        .reduce_shared(&stream, cfg, &data_dev, &mut partial_shared, N as u32)
        .expect("Shared kernel launch failed");
    let partial_shared_host = partial_shared.to_host_vec(&stream).unwrap();
    let sum_shared: f32 = partial_shared_host.iter().sum();
    let ok_shared = (sum_shared - expected).abs() < 1.0;
    println!(
        "  Sum = {:.6}  {}  (diff = {:.6})",
        sum_shared,
        if ok_shared { "✓ PASS" } else { "✗ FAIL" },
        (sum_shared - expected).abs()
    );

    // ----- Warp shuffle -----
    println!("\n--- 3. Warp Shuffle (register-to-register) ---");
    let mut partial_warp = DeviceBuffer::<f32>::zeroed(&stream, num_blocks).unwrap();
    module
        .reduce_warp_shuffle(&stream, cfg, &data_dev, &mut partial_warp, N as u32)
        .expect("Warp shuffle kernel launch failed");
    let partial_warp_host = partial_warp.to_host_vec(&stream).unwrap();
    let sum_warp: f32 = partial_warp_host.iter().sum();
    let ok_warp = (sum_warp - expected).abs() < 1.0;
    println!(
        "  Sum = {:.6}  {}  (diff = {:.6})",
        sum_warp,
        if ok_warp { "✓ PASS" } else { "✗ FAIL" },
        (sum_warp - expected).abs()
    );

    if ok_naive && ok_shared && ok_warp {
        println!("\n✓ SUCCESS: All three reduction strategies verified!");
        println!("  - Naïve:     adjacent pairing with warp divergence");
        println!("  - Shared:    tree reduction with contiguous threads");
        println!("  - Shuffle:   register-level warp primitives");
    } else {
        println!("\n✗ FAILED: Some reduction results incorrect.");
        std::process::exit(1);
    }
}
