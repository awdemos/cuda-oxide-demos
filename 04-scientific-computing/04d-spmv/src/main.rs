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
    pub fn spmv_csr(
        values: &[f32],
        col_indices: &[u32],
        row_ptr: &[u32],
        x: &[f32],
        mut y: DisjointSlice<f32>,
        n: u32,
    ) {
        let row = thread::blockIdx_x() as usize;
        if row >= n as usize {
            return;
        }
        let start = row_ptr[row] as usize;
        let end = row_ptr[row + 1] as usize;
        let mut sum = 0.0f32;
        for j in start..end {
            sum += values[j] * x[col_indices[j] as usize];
        }
        if let Some(e) = y.get_mut(thread::index_1d()) {
            *e = sum;
        }
    }
}

fn main() {
    println!("=== Sparse Matrix-Vector Multiply (CSR) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load kernel module");

    const N: usize = 256;
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_ptr = Vec::with_capacity(N + 1);
    let mut row_start = 0usize;
    for i in 0..N {
        row_ptr.push(row_start as u32);
        if i > 0 {
            values.push(0.5);
            col_indices.push((i - 1) as u32);
            row_start += 1;
        }
        values.push(1.0);
        col_indices.push(i as u32);
        row_start += 1;
        if i < N - 1 {
            values.push(0.5);
            col_indices.push((i + 1) as u32);
            row_start += 1;
        }
    }
    row_ptr.push(row_start as u32);

    let x: Vec<f32> = vec![1.0; N];

    let values_dev = DeviceBuffer::from_host(&stream, &values).unwrap();
    let col_dev = DeviceBuffer::from_host(&stream, &col_indices).unwrap();
    let row_ptr_dev = DeviceBuffer::from_host(&stream, &row_ptr).unwrap();
    let x_dev = DeviceBuffer::from_host(&stream, &x).unwrap();
    let mut y_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (N as u32, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    module
        .spmv_csr(&stream, cfg, &values_dev, &col_dev, &row_ptr_dev, &x_dev, &mut y_dev, N as u32)
        .expect("spmv_csr failed");

    let result = y_dev.to_host_vec(&stream).unwrap();
    let first = result[0];
    let last = result[N - 1];
    let middle = result[N / 2];
    println!("y[0]   = {:.4} (expected 1.5)", first);
    println!("y[128] = {:.4} (expected 2.0)", middle);
    println!("y[255] = {:.4} (expected 1.5)", last);
    let ok = (first - 1.5).abs() < 1e-4 && (middle - 2.0).abs() < 1e-4 && (last - 1.5).abs() < 1e-4;
    println!("  {}", if ok { "✓ PASS" } else { "✗ FAIL" });
}
