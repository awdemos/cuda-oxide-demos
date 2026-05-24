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
    pub fn rng_fill(mut out: DisjointSlice<f32>, n: u32) {
        let i = thread::index_1d().get() as u32;
        if i >= n {
            return;
        }
        let mut seed = (i + 1) * 2654435769u32;
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        let val = (seed >> 8) as f32 / 16777216.0f32;
        if let Some(e) = out.get_mut(thread::index_1d()) {
            *e = val;
        }
    }
}

fn main() {
    println!("=== GPU Random Number Generator ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load kernel module");

    const N: usize = 1024;
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let cfg = LaunchConfig::for_num_elems(N as u32);
    module
        .rng_fill((&stream).into(), cfg, &mut out_dev, N as u32)
        .expect("rng_fill failed");

    let vals = out_dev.to_host_vec(&stream).unwrap();
    let mean: f32 = vals.iter().sum::<f32>() / N as f32;
    let variance: f32 = vals.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / N as f32;

    println!("N = {}", N);
    println!("Mean:     {:.4}  (expected ~0.5)", mean);
    println!("Variance: {:.4}  (expected ~0.083)", variance);

    let mean_ok = (mean - 0.5).abs() < 0.1;
    let var_ok = (variance - 0.0833).abs() < 0.05;
    println!("  {}", if mean_ok && var_ok { "✓ PASS" } else { "✗ FAIL" });
}
