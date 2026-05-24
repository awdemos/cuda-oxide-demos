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
    pub fn all_reduce(data: &[f32], mut out: DisjointSlice<f32>, n: u32) {
        let i = thread::index_1d().get() as usize;
        if i >= n as usize {
            return;
        }
        if let Some(e) = out.get_mut(thread::index_1d()) {
            *e = data[i];
        }
    }
}

fn main() {
    println!("=== NCCL-Style All-Reduce (Single Rank Demo) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load kernel module");

    const N: usize = 1024;
    let data: Vec<f32> = (1..=N).map(|i| i as f32).collect();
    let data_dev = DeviceBuffer::from_host(&stream, &data).unwrap();
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let cfg = LaunchConfig::for_num_elems(N as u32);
    module
        .all_reduce((&stream).into(), cfg, &data_dev, &mut out_dev, N as u32)
        .expect("all_reduce failed");

    let result = out_dev.to_host_vec(&stream).unwrap();
    let ok = result.iter().enumerate().all(|(i, &v)| (v - data[i]).abs() < 1e-5);
    println!("First 5: {:?}", &result[..5]);
    println!("  {}", if ok { "✓ PASS" } else { "✗ FAIL" });
    println!("\n  In a multi-GPU setup, this would sum contributions from all ranks.");
}
