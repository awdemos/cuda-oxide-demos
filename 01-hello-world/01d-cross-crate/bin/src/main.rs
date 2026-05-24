/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cross-crate Kernel Demo — binary consumes kernels from a library crate
//!
//! The kernel functions are defined in the sibling `lib/` crate. This binary
//! imports them, and monomorphization happens at the use site.
//!
//! Build and run with:
//!   cargo oxide run --manifest-path 01-hello-world/01d-cross-crate/bin/Cargo.toml

use cross_crate_lib::kernels;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

fn main() {
    println!("=== Cross-Crate Kernel Demo ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load kernel module");

    const N: usize = 1024;
    let cfg = LaunchConfig::for_num_elems(N as u32);

    // Use scale::<f32> defined in the library crate
    println!("scale::<f32> from library crate (factor=4.0)");
    {
        let input: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
        module
            .scale::<f32>(&stream, cfg, 4.0f32, &input_dev, &mut out_dev)
            .expect("scale::<f32> failed");
        let result = out_dev.to_host_vec(&stream).unwrap();
        let ok = (0..N).all(|i| (result[i] - input[i] * 4.0).abs() < 1e-5);
        println!("  {}", if ok { "✓ PASS" } else { "✗ FAIL" });
    }

    // Use add::<f32> defined in the library crate
    println!("add::<f32> from library crate");
    {
        let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..N).map(|i| (i * 3) as f32).collect();
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

    println!("\n✓ Cross-crate kernel demo passed.");
}
