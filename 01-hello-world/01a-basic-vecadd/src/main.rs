/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Basic Vector Addition — simplest possible kernel
//!
//! Build and run with:
//!   cargo oxide run --manifest-path 01-hello-world/01a-basic-vecadd/Cargo.toml

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// Element-wise vector addition: c[i] = a[i] + b[i]
    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[idx_raw] + b[idx_raw];
        }
    }
}

fn main() {
    println!("=== Basic Vector Addition ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    println!("Input a[0..5] = {:?}", &a_host[..5]);
    println!("Input b[0..5] = {:?}", &b_host[..5]);

    let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load kernel module");
    module
        .vecadd(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &a_dev,
            &b_dev,
            &mut c_dev,
        )
        .expect("Kernel launch failed");

    let c_host = c_dev.to_host_vec(&stream).unwrap();

    println!("\nOutput c[0..5] = {:?}", &c_host[..5]);

    let errors: usize = (0..N)
        .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
        .count();

    if errors == 0 {
        println!("\n✓ SUCCESS: All {} elements correct!", N);
    } else {
        println!("\n✗ FAILED: {} errors", errors);
        std::process::exit(1);
    }
}
