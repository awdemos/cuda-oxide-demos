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
    pub fn stage1(input: &[f32], mut mid: DisjointSlice<f32>, n: u32) {
        let i = thread::index_1d().get() as usize;
        if i >= n as usize {
            return;
        }
        if let Some(e) = mid.get_mut(thread::index_1d()) {
            *e = input[i] * 2.0 + 1.0;
        }
    }

    #[kernel]
    pub fn stage2(mid: &[f32], mut out: DisjointSlice<f32>, n: u32) {
        let i = thread::index_1d().get() as usize;
        if i >= n as usize {
            return;
        }
        if let Some(e) = out.get_mut(thread::index_1d()) {
            *e = mid[i] * mid[i];
        }
    }
}

fn main() {
    println!("=== Pipeline-Parallel Computation ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load kernel module");

    const N: usize = 1024;
    let input: Vec<f32> = (0..N).map(|i| i as f32 * 0.01).collect();
    let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
    let mut mid_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let cfg = LaunchConfig::for_num_elems(N as u32);

    module
        .stage1((&stream).into(), cfg, &input_dev, &mut mid_dev, N as u32)
        .expect("stage1 failed");
    module
        .stage2((&stream).into(), cfg, &mid_dev, &mut out_dev, N as u32)
        .expect("stage2 failed");

    let result = out_dev.to_host_vec(&stream).unwrap();
    let expected = |i: f32| (i * 0.01 * 2.0 + 1.0).powi(2);
    let ok = result.iter().enumerate().all(|(i, &v)| (v - expected(i as f32)).abs() < 1e-3);
    println!("First 5: {:?}", &result[..5]);
    println!("  {}", if ok { "✓ PASS" } else { "✗ FAIL" });
}
