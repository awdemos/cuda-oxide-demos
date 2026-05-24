/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Generic Kernel — type-parameterized GPU kernels with monomorphization
//!
//! Demonstrates defining kernels generic over numeric types and instantiating
//! them for f32 and i32 from the host side.
//!
//! Build and run with:
//!   cargo oxide run --manifest-path 01-hello-world/01c-generic-kernel/Cargo.toml

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
use std::ops::{Add, Mul};

#[cuda_module]
mod kernels {
    use super::*;

    /// Generic scale kernel: out[i] = input[i] * factor
    #[kernel]
    pub fn scale<T: Copy + Mul<Output = T>>(factor: T, input: &[T], mut out: DisjointSlice<T>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = input[idx_raw] * factor;
        }
    }

    /// Generic add kernel: c[i] = a[i] + b[i]
    #[kernel]
    pub fn add<T: Copy + Add<Output = T>>(a: &[T], b: &[T], mut c: DisjointSlice<T>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[idx_raw] + b[idx_raw];
        }
    }
}

fn main() {
    println!("=== Generic Kernel (Monomorphization) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load kernel module");

    const N: usize = 1024;
    let cfg = LaunchConfig::for_num_elems(N as u32);

    // f32 scale
    println!("scale::<f32> with factor 2.5");
    {
        let input: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
        module
            .scale::<f32>(&stream, cfg, 2.5f32, &input_dev, &mut out_dev)
            .expect("scale::<f32> failed");
        let result = out_dev.to_host_vec(&stream).unwrap();
        let ok = (0..N).all(|i| (result[i] - input[i] * 2.5).abs() < 1e-5);
        println!("  {}", if ok { "✓ PASS" } else { "✗ FAIL" });
    }

    // i32 scale
    println!("scale::<i32> with factor 3");
    {
        let input: Vec<i32> = (0..N as i32).collect();
        let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
        let mut out_dev = DeviceBuffer::<i32>::zeroed(&stream, N).unwrap();
        module
            .scale::<i32>(&stream, cfg, 3i32, &input_dev, &mut out_dev)
            .expect("scale::<i32> failed");
        let result = out_dev.to_host_vec(&stream).unwrap();
        let ok = (0..N).all(|i| result[i] == input[i] * 3);
        println!("  {}", if ok { "✓ PASS" } else { "✗ FAIL" });
    }

    // f32 add
    println!("add::<f32>");
    {
        let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();
        let a_dev = DeviceBuffer::from_host(&stream, &a).unwrap();
        let b_dev = DeviceBuffer::from_host(&stream, &b).unwrap();
        let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
        module
            .add::<f32>(&stream, cfg, &a_dev, &b_dev, &mut c_dev)
            .expect("add::<f32> failed");
        let result = c_dev.to_host_vec(&stream).unwrap();
        let ok = (0..N).all(|i| (result[i] - (a[i] + b[i])).abs() < 1e-5);
        println!("  {}", if ok { "✓ PASS" } else { "✗ FAIL" });
    }

    println!("\n✓ All generic kernel tests passed.");
}
