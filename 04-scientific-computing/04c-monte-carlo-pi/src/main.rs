/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn monte_carlo_pi(mut hits: DisjointSlice<u32>, n_per_thread: u32) {
        let tid = thread::index_1d().get() as u32;
        let mut seed = (tid + 1) * 2654435769u32;
        let mut local_hits = 0u32;

        for _ in 0..n_per_thread {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let x = seed as f32 / u32::MAX as f32;
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let y = seed as f32 / u32::MAX as f32;
            if x * x + y * y < 1.0 {
                local_hits += 1;
            }
        }

        if let Some(e) = hits.get_mut(thread::index_1d()) {
            *e = local_hits;
        }
    }
}

fn main() {
    println!("=== Monte Carlo Pi ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load kernel module");

    const BLOCKS: u32 = 1024;
    const THREADS: u32 = 256;
    const SAMPLES_PER_THREAD: u32 = 1000;
    let total_threads = BLOCKS * THREADS;
    let total_samples = total_threads * SAMPLES_PER_THREAD;

    let mut hits_dev = DeviceBuffer::<u32>::zeroed(&stream, total_threads as usize).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (BLOCKS, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    };

    module
        .monte_carlo_pi(&stream, cfg, &mut hits_dev, SAMPLES_PER_THREAD)
        .expect("monte_carlo_pi failed");

    let hits_host = hits_dev.to_host_vec(&stream).unwrap();
    let total_hits: u64 = hits_host.iter().map(|&h| h as u64).sum();
    let pi_estimate = 4.0 * total_hits as f64 / total_samples as f64;

    println!("Samples: {}", total_samples);
    println!("Hits: {}", total_hits);
    println!("Pi ≈ {:.6}", pi_estimate);
    println!("Error: {:.6}", (pi_estimate - std::f64::consts::PI).abs());
    println!(
        "  {}",
        if (pi_estimate - std::f64::consts::PI).abs() < 0.1 {
            "✓ PASS"
        } else {
            "✗ FAIL (error too large)"
        }
    );
}
