/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};

const BLOCK: usize = 256;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn dot(x: &[f32], y: &[f32], mut result: DisjointSlice<f32>, n: u32) {
        static mut s: SharedArray<f32, BLOCK> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let gid = thread::index_1d().get() as usize;

        unsafe { s[tid] = if gid < n as usize { x[gid] * y[gid] } else { 0.0 }; }
        thread::sync_threads();

        let mut stride = BLOCK / 2;
        while stride > 0 {
            if tid < stride {
                unsafe { s[tid] += s[tid + stride]; }
            }
            thread::sync_threads();
            stride /= 2;
        }

        if tid == 0 {
            let bid = thread::blockIdx_x() as usize;
            if let Some(e) = result.get_mut(thread::index_1d()) {
                *e = unsafe { s[0] };
            }
        }
    }

    #[kernel]
    pub fn axpy(mut y: DisjointSlice<f32>, a: f32, x: &[f32], n: u32) {
        let i = thread::index_1d().get() as usize;
        if i >= n as usize {
            return;
        }
        if let Some(e) = y.get_mut(thread::index_1d()) {
            *e = y.get(thread::index_1d()).unwrap() + a * x[i];
        }
    }

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

    #[kernel]
    pub fn scale(mut x: DisjointSlice<f32>, s: f32, n: u32) {
        let i = thread::index_1d().get() as usize;
        if i >= n as usize {
            return;
        }
        if let Some(e) = x.get_mut(thread::index_1d()) {
            *e = x.get(thread::index_1d()).unwrap() * s;
        }
    }
}

fn main() {
    println!("=== Conjugate Gradient Solver ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load kernel module");

    const N: usize = 64;
    let mut values = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_ptr = Vec::with_capacity(N + 1);
    let mut row_start = 0usize;
    for i in 0..N {
        row_ptr.push(row_start as u32);
        if i > 0 {
            values.push(-1.0);
            col_indices.push((i - 1) as u32);
            row_start += 1;
        }
        values.push(2.0);
        col_indices.push(i as u32);
        row_start += 1;
        if i < N - 1 {
            values.push(-1.0);
            col_indices.push((i + 1) as u32);
            row_start += 1;
        }
    }
    row_ptr.push(row_start as u32);

    let b = vec![1.0f32; N];
    let val_dev = DeviceBuffer::from_host(&stream, &values).unwrap();
    let col_dev = DeviceBuffer::from_host(&stream, &col_indices).unwrap();
    let row_dev = DeviceBuffer::from_host(&stream, &row_ptr).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b).unwrap();

    let mut x_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let mut r_dev = b_dev.clone();
    let mut p_dev = r_dev.clone();
    let mut ap_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let mut partial = DeviceBuffer::<f32>::zeroed(&stream, 1).unwrap();

    let cfg1 = LaunchConfig { grid_dim: (N as u32, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
    let cfg_block = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (BLOCK as u32, 1, 1), shared_mem_bytes: 0 };

    let mut rdotr_old = 1.0f32;

    for iter in 0..N {
        module.spmv_csr((&stream).into(), cfg1, &val_dev, &col_dev, &row_dev, &p_dev, &mut ap_dev, N as u32)
            .expect("spmv failed");

        module.dot((&stream).into(), cfg_block, &p_dev, &ap_dev, &mut partial, N as u32)
            .expect("dot failed");
        let p_ap = partial.to_host_vec(&stream).unwrap()[0];
        let alpha = rdotr_old / p_ap;

        module.axpy((&stream).into(), LaunchConfig::for_num_elems(N as u32), x_dev.clone(), alpha, &p_dev, N as u32)
            .expect("axpy x failed");
        module.axpy((&stream).into(), LaunchConfig::for_num_elems(N as u32), r_dev.clone(), -alpha, &ap_dev, N as u32)
            .expect("axpy r failed");

        module.dot((&stream).into(), cfg_block, &r_dev, &r_dev, &mut partial, N as u32)
            .expect("dot r failed");
        let rdotr_new = partial.to_host_vec(&stream).unwrap()[0];

        if rdotr_new.sqrt() < 1e-6 {
            println!("Converged at iteration {}", iter);
            break;
        }

        let beta = rdotr_new / rdotr_old;
        module.scale((&stream).into(), LaunchConfig::for_num_elems(N as u32), p_dev.clone(), beta, N as u32)
            .expect("scale p failed");
        module.axpy((&stream).into(), LaunchConfig::for_num_elems(N as u32), p_dev.clone(), 1.0, &r_dev, N as u32)
            .expect("axpy p failed");

        rdotr_old = rdotr_new;
    }

    let x = x_dev.to_host_vec(&stream).unwrap();
    println!("Solution x[0..4]: {:?}", &x[..4]);
    println!("  {} (expected ~N/2)", if x[N/2] > 15.0 && x[N/2] < 17.0 { "✓ PASS" } else { "✗ FAIL" });
}
