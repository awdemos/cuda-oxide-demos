/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Closure Arguments — generic kernel that accepts a host closure
//!
//! Demonstrates passing closures (with captures) into GPU kernels.
//! Captured variables are passed as a single byval struct.
//!
//! Build and run with:
//!   cargo oxide run --manifest-path 01-hello-world/01b-closure-params/Cargo.toml

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// Generic map kernel — applies a function to each element.
    #[kernel]
    pub fn map<T: Copy, F: Fn(T) -> T + Copy>(f: F, input: &[T], mut out: DisjointSlice<T>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = f(input[idx_raw]);
        }
    }
}

fn main() {
    println!("=== Closure-Parameterized Kernel ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let input: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load kernel module");
    let cfg = LaunchConfig::for_num_elems(N as u32);

    // Test 1: Single capture
    println!("Test 1: Single capture (scale by 2.5)");
    {
        let factor = 2.5f32;
        out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
        module
            .map::<f32, _>(&stream, cfg, move |x: f32| x * factor, &input_dev, &mut out_dev)
            .expect("Kernel launch failed");
        let result = out_dev.to_host_vec(&stream).unwrap();
        let ok = (0..N).all(|i| (result[i] - input[i] * factor).abs() < 1e-5);
        println!("  {}", if ok { "✓ PASS" } else { "✗ FAIL" });
    }

    // Test 2: Multiple captures
    println!("Test 2: Multiple captures (polynomial: a*x² + b*x + c)");
    {
        let a = 0.5f32;
        let b = 2.0f32;
        let c = 1.0f32;
        out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
        module
            .map::<f32, _>(
                &stream,
                cfg,
                move |x: f32| a * x * x + b * x + c,
                &input_dev,
                &mut out_dev,
            )
            .expect("Kernel launch failed");
        let result = out_dev.to_host_vec(&stream).unwrap();
        let ok = (0..N).all(|i| {
            let expected = a * input[i] * input[i] + b * input[i] + c;
            (result[i] - expected).abs() < 1e-3
        });
        println!("  {}", if ok { "✓ PASS" } else { "✗ FAIL" });
    }

    // Test 3: Zero captures (inline constant)
    println!("Test 3: Zero captures (double each element)");
    {
        out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
        module
            .map::<f32, _>(&stream, cfg, |x: f32| x * 2.0, &input_dev, &mut out_dev)
            .expect("Kernel launch failed");
        let result = out_dev.to_host_vec(&stream).unwrap();
        let ok = (0..N).all(|i| (result[i] - input[i] * 2.0).abs() < 1e-5);
        println!("  {}", if ok { "✓ PASS" } else { "✗ FAIL" });
    }

    println!("\n✓ All closure tests completed.");
}
