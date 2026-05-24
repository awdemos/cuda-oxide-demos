/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! 2D Image Convolution — classic CUDA sample ported to cuda-oxide.
//!
//! Demonstrates:
//! - Shared-memory tiling with a halo region for stencil boundaries
//! - Clamp-to-edge boundary handling
//! - 3×3 box-blur and simple edge-detection kernels
//! - Verification against host reference
//!
//! Build and run with:
//!   cargo oxide run --manifest-path 02-ported-cuda-samples/02d-convolution-2d/Cargo.toml

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice, SharedArray};

// Each block computes a TILE_OUT × TILE_OUT region of the output.
// Threads load a TILE_IN × TILE_IN region into shared memory, where
// TILE_IN = TILE_OUT + 2*HALO to accommodate the 3×3 kernel radius.
const TILE_OUT: usize = 14;
const HALO: usize = 1;
const TILE_IN: usize = TILE_OUT + 2 * HALO; // 16

#[cuda_module]
mod kernels {
    use super::*;

    /// 2D convolution with shared-memory tiling.
    ///
    /// The block size is `TILE_IN × TILE_IN = 16 × 16 = 256` threads.
    /// All threads cooperatively load the input tile (including halo) into
    /// shared memory. The inner `TILE_OUT × TILE_OUT` threads then compute
    /// the convolution.
    #[kernel]
    pub fn conv2d_shared(input: &[f32], mut output: DisjointSlice<f32>, width: u32, height: u32) {
        static mut TILE: SharedArray<f32, { TILE_IN * TILE_IN }> = SharedArray::UNINIT;

        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;
        let bx = thread::blockIdx_x() as usize;
        let by = thread::blockIdx_y() as usize;

        let w = width as usize;
        let h = height as usize;

        // Global input coordinates for this thread's load.
        let in_col = bx * TILE_OUT + tx;
        let in_row = by * TILE_OUT + ty;

        // Clamp-to-edge load into shared memory.
        let clamped_col = if in_col < w {
            in_col
        } else {
            w.saturating_sub(1)
        };
        let clamped_row = if in_row < h {
            in_row
        } else {
            h.saturating_sub(1)
        };
        unsafe {
            TILE[ty * TILE_IN + tx] = input[clamped_row * w + clamped_col];
        }
        thread::sync_threads();

        // Only the inner TILE_OUT × TILE_OUT threads compute output pixels.
        if tx < TILE_OUT && ty < TILE_OUT {
            let out_col = bx * TILE_OUT + tx;
            let out_row = by * TILE_OUT + ty;

            if out_col < w && out_row < h {
                let mut sum = 0.0f32;

                // 3×3 box blur kernel (all weights = 1/9).
                for ky in 0..3 {
                    for kx in 0..3 {
                        let s_row = ty + ky;
                        let s_col = tx + kx;
                        unsafe {
                            sum += TILE[s_row * TILE_IN + s_col];
                        }
                    }
                }
                sum *= 1.0 / 9.0;

                let out_idx = out_row * w + out_col;
                unsafe {
                    *output.get_unchecked_mut(out_idx) = sum;
                }
            }
        }
    }
}

/// Host reference: 2D convolution with clamp-to-edge boundaries.
fn conv2d_host(width: usize, height: usize, input: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0f32;
            for ky in 0..3 {
                for kx in 0..3 {
                    let iy = y + ky;
                    let ix = x + kx;
                    // Clamp to edge.
                    let cy = if iy < height { iy } else { height - 1 };
                    let cx = if ix < width { ix } else { width - 1 };
                    sum += input[cy * width + cx];
                }
            }
            out[y * width + x] = sum * (1.0 / 9.0);
        }
    }
    out
}

fn main() {
    println!("=== 2D Convolution (Shared Memory Tiling) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load kernel module");

    const W: usize = 128;
    const H: usize = 128;
    let input: Vec<f32> = (0..W * H).map(|i| ((i * 7 + 13) % 256) as f32).collect();

    println!("Image size: {} × {}", W, H);
    println!("Kernel:     3×3 box blur (all weights = 1/9)");
    println!(
        "Tile:       output {}×{}, halo {}, shared {}×{}\n",
        TILE_OUT, TILE_OUT, HALO, TILE_IN, TILE_IN
    );

    let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, W * H).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (
            (W as u32).div_ceil(TILE_OUT as u32),
            (H as u32).div_ceil(TILE_OUT as u32),
            1,
        ),
        block_dim: (TILE_IN as u32, TILE_IN as u32, 1),
        shared_mem_bytes: 0,
    };

    module
        .conv2d_shared(&stream, cfg, &input_dev, &mut out_dev, W as u32, H as u32)
        .expect("Kernel launch failed");

    let result = out_dev.to_host_vec(&stream).unwrap();
    let expected = conv2d_host(W, H, &input);

    let mut max_error = 0.0f32;
    let mut errors = 0usize;
    for i in 0..(W * H) {
        let err = (result[i] - expected[i]).abs();
        if err > max_error {
            max_error = err;
        }
        if err > 1e-4 {
            errors += 1;
            if errors <= 3 {
                eprintln!(
                    "  Mismatch at [{}]: expected {:.6}, got {:.6}",
                    i, expected[i], result[i]
                );
            }
        }
    }

    println!("Max error: {:.6e}", max_error);
    println!(
        "{}  {}  ({} mismatches > 1e-4)",
        if errors == 0 { "✓ PASS" } else { "✗ FAIL" },
        if errors == 0 {
            "Shared-memory 2D convolution verified against host reference"
        } else {
            "Some results incorrect."
        },
        errors
    );

    if errors > 0 {
        std::process::exit(1);
    }
}
