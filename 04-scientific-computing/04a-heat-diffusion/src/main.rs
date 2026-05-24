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
    pub fn heat_step(
        grid: &[f32],
        mut next: DisjointSlice<f32>,
        width: u32,
        height: u32,
        alpha: f32,
        dt: f32,
    ) {
        let x = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let y = thread::blockIdx_y() * thread::blockDim_y() + thread::threadIdx_y();

        if x >= width || y >= height {
            return;
        }

        let idx = (y * width + x) as usize;
        let factor = alpha * dt;

        let left = if x > 0 { grid[idx - 1] } else { grid[idx] };
        let right = if x < width - 1 { grid[idx + 1] } else { grid[idx] };
        let top = if y > 0 { grid[idx - width as usize] } else { grid[idx] };
        let bottom = if y < height - 1 { grid[idx + width as usize] } else { grid[idx] };

        let idx_2d = unsafe { thread::index_2d_runtime(width as usize).unwrap() };
        unsafe {
            *next.get_unchecked_mut(idx_2d.get()) =
                grid[idx] + factor * (left + right + top + bottom - 4.0 * grid[idx]);
        }
    }
}

fn main() {
    println!("=== 2D Heat Diffusion ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load kernel module");

    const W: usize = 256;
    const H: usize = 256;
    let mut grid = vec![0.0f32; W * H];
    for y in 96..160 {
        for x in 96..160 {
            grid[y * W + x] = 1.0;
        }
    }

    let mut grid_dev = DeviceBuffer::from_host(&stream, &grid).unwrap();
    let mut next_dev = DeviceBuffer::<f32>::zeroed(&stream, W * H).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (16, 16, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };

    const ALPHA: f32 = 0.01;
    const DT: f32 = 0.1;
    const STEPS: usize = 100;

    for _step in 0..STEPS {
        module
            .heat_step(
                &stream, cfg,
                &grid_dev, &mut next_dev,
                W as u32, H as u32, ALPHA, DT,
            )
            .expect("heat_step failed");
        std::mem::swap(&mut grid_dev, &mut next_dev);
    }

    let result = grid_dev.to_host_vec(&stream).unwrap();

    let center = result[128 * W + 128];
    let corner = result[0];
    println!("Center temp after {} steps: {:.6}", STEPS, center);
    println!("Corner temp after {} steps: {:.6}", STEPS, corner);
    println!("  {}", if center > corner && center < 1.0 && corner > 0.0 { "✓ PASS" } else { "✗ FAIL" });
}
