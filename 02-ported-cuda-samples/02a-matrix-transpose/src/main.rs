/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Matrix Transpose — classic CUDA sample ported to cuda-oxide.
//!
//! Demonstrates:
//! - Naïve global-memory transpose (coalesced reads, strided writes)
//! - Shared-memory tiled transpose with bank-conflict avoidance (+1 padding)
//! - Verification against host transpose
//!
//! Build and run with:
//!   cargo oxide run --manifest-path 02-ported-cuda-samples/02a-matrix-transpose/Cargo.toml

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};

const TILE_DIM: usize = 32;

#[cuda_module]
mod kernels {
    use super::*;

    /// Naïve transpose: each thread reads one element and writes to the
    /// transposed position directly in global memory. Reads are coalesced,
    /// but writes are strided (uncoalesced) — poor performance.
    #[kernel]
    pub fn transpose_naive(input: &[f32], mut output: DisjointSlice<f32>, width: u32, height: u32) {
        let x = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
        let y = thread::blockIdx_y() * thread::blockDim_y() + thread::threadIdx_y();

        if x < width && y < height {
            let in_idx = (y * width + x) as usize;
            let out_idx = (x * height + y) as usize;
            unsafe {
                *output.get_unchecked_mut(out_idx) = input[in_idx];
            }
        }
    }

    /// Shared-memory tiled transpose with bank-conflict avoidance.
    ///
    /// The shared tile is sized `TILE_DIM × (TILE_DIM + 1)` so that
    /// columns of the tile (which become rows after the transpose)
    /// map to different shared-memory banks.
    #[kernel]
    pub fn transpose_shared(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        width: u32,
        height: u32,
    ) {
        // Padded tile: +1 in the X dimension avoids bank conflicts on the
        // transposed read (adjacent threads in X read different banks).
        static mut TILE: SharedArray<f32, { TILE_DIM * (TILE_DIM + 1) }> = SharedArray::UNINIT;

        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let bx = thread::blockIdx_x() as usize;
        let by = thread::blockIdx_y() as usize;

        let x = bx * TILE_DIM + tx;
        let y = by * TILE_DIM + ty;
        let w = width as usize;
        let h = height as usize;

        // Load tile from global memory into shared memory (coalesced read).
        if x < w && y < h {
            unsafe {
                TILE[ty * (TILE_DIM + 1) + tx] = input[y * w + x];
            }
        }

        thread::sync_threads();

        // Compute transposed output position.
        // The block at input (bx, by) is written to output block (by, bx).
        // Thread (tx, ty) reads shared[tx][ty] and writes to the transposed
        // coordinates — swapping block indices and swapping thread indices.
        let x_out = by * TILE_DIM + tx;
        let y_out = bx * TILE_DIM + ty;

        if x_out < h && y_out < w {
            unsafe {
                let val = TILE[tx * (TILE_DIM + 1) + ty];
                let out_idx = y_out * h + x_out;
                *output.get_unchecked_mut(out_idx) = val;
            }
        }
    }
}

fn transpose_host(width: usize, height: usize, input: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            out[x * height + y] = input[y * width + x];
        }
    }
    out
}

fn main() {
    println!("=== Matrix Transpose ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load kernel module");

    const W: usize = 512;
    const H: usize = 768;
    let input: Vec<f32> = (0..W * H).map(|i| (i % 100) as f32 + 0.5).collect();

    println!("Matrix size: {} × {} ({} elements)", W, H, W * H);
    println!(
        "Tile size:   {} × {} (padded to {} × {})\n",
        TILE_DIM,
        TILE_DIM,
        TILE_DIM,
        TILE_DIM + 1
    );

    let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
    let mut out_naive_dev = DeviceBuffer::<f32>::zeroed(&stream, W * H).unwrap();
    let mut out_shared_dev = DeviceBuffer::<f32>::zeroed(&stream, W * H).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (
            (W as u32).div_ceil(TILE_DIM as u32),
            (H as u32).div_ceil(TILE_DIM as u32),
            1,
        ),
        block_dim: (TILE_DIM as u32, TILE_DIM as u32, 1),
        shared_mem_bytes: 0,
    };

    // ----- Naïve transpose -----
    println!("Running naïve transpose...");
    module
        .transpose_naive(
            &stream,
            cfg,
            &input_dev,
            &mut out_naive_dev,
            W as u32,
            H as u32,
        )
        .expect("Naïve kernel launch failed");

    let result_naive = out_naive_dev.to_host_vec(&stream).unwrap();
    let expected = transpose_host(W, H, &input);
    let mut errors_naive = 0usize;
    for i in 0..(W * H) {
        if (result_naive[i] - expected[i]).abs() > 1e-5 {
            errors_naive += 1;
            if errors_naive <= 3 {
                eprintln!(
                    "  Naïve mismatch at [{}]: expected {}, got {}",
                    i, expected[i], result_naive[i]
                );
            }
        }
    }
    println!(
        "  Naïve: {} errors  {}",
        errors_naive,
        if errors_naive == 0 {
            "✓ PASS"
        } else {
            "✗ FAIL"
        }
    );

    // ----- Shared-memory transpose -----
    println!("Running shared-memory transpose...");
    module
        .transpose_shared(
            &stream,
            cfg,
            &input_dev,
            &mut out_shared_dev,
            W as u32,
            H as u32,
        )
        .expect("Shared kernel launch failed");

    let result_shared = out_shared_dev.to_host_vec(&stream).unwrap();
    let mut errors_shared = 0usize;
    for i in 0..(W * H) {
        if (result_shared[i] - expected[i]).abs() > 1e-5 {
            errors_shared += 1;
            if errors_shared <= 3 {
                eprintln!(
                    "  Shared mismatch at [{}]: expected {}, got {}",
                    i, expected[i], result_shared[i]
                );
            }
        }
    }
    println!(
        "  Shared: {} errors  {}",
        errors_shared,
        if errors_shared == 0 {
            "✓ PASS"
        } else {
            "✗ FAIL"
        }
    );

    if errors_naive == 0 && errors_shared == 0 {
        println!("\n✓ SUCCESS: Both transpose variants verified!");
    } else {
        println!("\n✗ FAILED: Some results incorrect.");
        std::process::exit(1);
    }
}
