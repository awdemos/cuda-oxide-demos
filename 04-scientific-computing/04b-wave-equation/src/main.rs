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
    pub fn wave_step(
        u: &[f32],
        mut unext: DisjointSlice<f32>,
        n: u32,
        c: f32,
        dt: f32,
        dx: f32,
    ) {
        let i = thread::index_1d().get() as usize;
        if i >= n as usize {
            return;
        }
        if i == 0 || i == n as usize - 1 {
            if let Some(e) = unext.get_mut(thread::index_1d()) {
                *e = 0.0;
            }
            return;
        }
        let coeff = (c * dt / dx) * (c * dt / dx);
        if let Some(e) = unext.get_mut(thread::index_1d()) {
            *e = u[i] + coeff * (u[i + 1] - 2.0 * u[i] + u[i - 1]);
        }
    }
}

fn main() {
    println!("=== 1D Wave Equation ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load kernel module");

    const N: usize = 1024;
    let mut u = vec![0.0f32; N];
    for i in 0..N {
        let x = (i as f64 - 512.0) / 50.0;
        u[i] = (-0.5 * x * x).exp() as f32;
    }

    let mut u_dev = DeviceBuffer::from_host(&stream, &u).unwrap();
    let mut unext_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let cfg = LaunchConfig::for_num_elems(N as u32);
    const C: f32 = 1.0;
    const DT: f32 = 0.01;
    const DX: f32 = 0.01;
    const STEPS: usize = 200;

    for _step in 0..STEPS {
        module
            .wave_step(&stream, cfg, &u_dev, &mut unext_dev, N as u32, C, DT, DX)
            .expect("wave_step failed");
        std::mem::swap(&mut u_dev, &mut unext_dev);
    }

    let result = u_dev.to_host_vec(&stream).unwrap();
    let center = result[N / 2];
    let spread = result.iter().filter(|&&v| v.abs() > 0.01).count();
    println!("Center amplitude after {} steps: {:.6}", STEPS, center);
    println!("Nonzero elements (|v|>0.01): {}", spread);
    println!("  {}", if center.abs() > 0.0 && spread < N { "✓ PASS" } else { "✗ FAIL" });
}
