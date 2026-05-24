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
    pub fn device_select(data: &[f32], mut out: DisjointSlice<f32>, device_id: u32, n: u32) {
        let i = thread::index_1d().get() as usize;
        if i >= n as usize {
            return;
        }
        if let Some(e) = out.get_mut(thread::index_1d()) {
            *e = data[i] + device_id as f32;
        }
    }
}

fn main() {
    println!("=== Peer Access / Multi-GPU Identity ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load kernel module");

    const N: usize = 1024;
    let data: Vec<f32> = (0..N).map(|i| i as f32 * 0.001).collect();
    let data_dev = DeviceBuffer::from_host(&stream, &data).unwrap();
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let cfg = LaunchConfig::for_num_elems(N as u32);
    module
        .device_select((&stream).into(), cfg, &data_dev, &mut out_dev, 0, N as u32)
        .expect("device_select failed");

    let result = out_dev.to_host_vec(&stream).unwrap();
    let ok = (0..N).all(|i| (result[i] - (data[i] + 0.0)).abs() < 1e-5);
    println!("First 5 elements: {:?}", &result[..5]);
    println!("  {}", if ok { "✓ PASS" } else { "✗ FAIL" });
}
