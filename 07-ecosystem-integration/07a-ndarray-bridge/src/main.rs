/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
use ndarray::Array2;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn scale(data: &[f32], mut out: DisjointSlice<f32>, n: u32) {
        let i = thread::index_1d().get() as usize;
        if i >= n as usize {
            return;
        }
        if let Some(e) = out.get_mut(thread::index_1d()) {
            *e = data[i] * 2.0;
        }
    }
}

fn main() {
    println!("=== ndarray Bridge ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load kernel module");

    let arr = Array2::eye(6);
    println!("Before:\n{}", arr);

    let flat: Vec<f32> = arr.iter().cloned().collect();
    let data_dev = DeviceBuffer::from_host(&stream, &flat).unwrap();
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, 36).unwrap();

    let cfg = LaunchConfig::for_num_elems(36);
    module
        .scale((&stream).into(), cfg, &data_dev, &mut out_dev, 36)
        .expect("scale failed");

    let result = out_dev.to_host_vec(&stream).unwrap();
    let scaled = Array2::from_shape_vec((6, 6), result).expect("reshape");
    println!("\nAfter (scaled x2):\n{}", scaled);
    println!("  ✓ PASS");
}
